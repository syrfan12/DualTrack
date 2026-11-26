"""Behavioral checks for the 2D trajectory and encoder-freezing boundary."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.models.mdpi_fusion import (
    MDPIFusionModel,
    PAIR_INDICES,
    build_mdpi_fusion,
    pair_mse_loss,
    pair_translations,
    steps_to_positions,
    targets_to_pairs,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(1, 4)
        self.bn = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.8)
        self.projection = nn.Linear(4, 3)

    def forward(self, images, indices=None):
        batch, length = images.shape[:2]
        x = self.image(images.mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1))
        x = self.bn(x.reshape(-1, 4)).reshape(batch, length, 4)
        return self.projection(self.dropout(x))


class TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, local, encoder_hidden_states):
        return self.fc(self.dropout(local + encoder_hidden_states.mean(dim=1, keepdim=True)))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 6)

    def forward(self, inputs):
        return self.fc(inputs)[:, 1:, :]


class TinySampler(nn.Module):
    def forward(self, images, features):
        return torch.arange(0, images.shape[1], 2, device=images.device)[None].expand(images.shape[0], -1)


class TinyDualTrack(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = TinyEncoder()
        self.global_encoder = TinyEncoder()
        self.fusion_module = TinyFusion()
        self.head = TinyHead()
        self.sampler = TinySampler()


class TrajectoryMathTests(unittest.TestCase):
    def test_signed_nonuniform_steps_and_fixed_pair_direction(self):
        steps = torch.tensor([[1., -2.], [-3., 4.], [2., 1.], [-2., -1.], [4., 5.],
                              [0., -3.], [-1., 2.], [3., 0.], [-4., 1.]])
        positions = steps_to_positions(steps)
        self.assertEqual(tuple(positions.shape), (10, 2))
        torch.testing.assert_close(positions[0], torch.zeros(2))
        torch.testing.assert_close(positions[2], torch.tensor([-2., 2.]))
        pairs = pair_translations(steps)
        self.assertEqual(tuple(pairs.shape), (45, 2))
        self.assertEqual(PAIR_INDICES.tolist()[:6], [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
        for slot, (i, j) in enumerate(PAIR_INDICES.tolist()):
            torch.testing.assert_close(pairs[slot], steps[i:j].sum(dim=0))
        torch.testing.assert_close(pairs, targets_to_pairs(positions + 123.))

    def test_pair_loss_retains_gradients_through_integration(self):
        steps = torch.ones(2, 9, 2, requires_grad=True)
        loss = pair_mse_loss(steps, torch.zeros(2, 10, 2))
        loss.backward()
        self.assertTrue(torch.isfinite(steps.grad).all())
        self.assertTrue((steps.grad != 0).all())

    def test_bad_shapes_and_broadcasting_are_rejected(self):
        for value in (torch.ones(9, 3), torch.ones(0, 2), torch.ones(9, 2, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                steps_to_positions(value)
        with self.assertRaises(ValueError):
            pair_translations(torch.zeros(8, 2))
        with self.assertRaises(ValueError):
            pair_mse_loss(torch.ones(2, 9, 2), torch.zeros(1, 10, 2))


class FusionBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.original = TinyDualTrack()
        self.old_weight = self.original.head.fc.weight.detach().clone()
        self.old_bias = self.original.head.fc.bias.detach().clone()
        self.model = MDPIFusionModel(self.original)
        self.images = torch.rand(2, 10, 1, 8, 8)

    def test_head_copies_only_measured_translation_rows(self):
        torch.testing.assert_close(self.model.dualtrack.head.fc.weight, self.old_weight[[1, 2]])
        torch.testing.assert_close(self.model.dualtrack.head.fc.bias, self.old_bias[[1, 2]])
        self.assertEqual(tuple(self.model(self.images, self.images).shape), (2, 9, 2))

    def test_modes_optimizer_and_parameter_scope_are_explicit(self):
        self.model.eval()
        self.assertFalse(self.model.dualtrack.fusion_module.training)
        self.model.train()
        for module in (self.model.dualtrack.local_encoder, self.model.dualtrack.global_encoder):
            self.assertFalse(module.training)
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        self.assertTrue(self.model.dualtrack.fusion_module.training)
        groups = self.model.trainable_parameter_groups(0.001, 0.01)
        ids = {id(p) for group in groups for p in group['params']}
        self.assertEqual(ids, {id(p) for p in self.model.parameters() if p.requires_grad})
        self.assertEqual([g['lr'] for g in groups], [0.001, 0.01])
        self.model.dualtrack.local_encoder.projection.weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            self.model.trainable_parameter_groups()

    def test_optimizer_updates_fusion_head_but_no_encoder_state(self):
        before = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}
        optimizer = torch.optim.AdamW(self.model.trainable_parameter_groups(0.01, 0.01))
        optimizer.zero_grad()
        predictions = self.model(self.images, self.images)
        loss = pair_mse_loss(predictions, torch.zeros(2, 10, 2))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if '.local_encoder.' in name or '.global_encoder.' in name:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        changed = set()
        for name, tensor in self.model.state_dict().items():
            if '.local_encoder.' in name or '.global_encoder.' in name:
                self.assertTrue(torch.equal(before[name], tensor), name)
            elif not torch.equal(before[name], tensor):
                changed.add(name)
        self.assertTrue(any('.fusion_module.' in name for name in changed))
        self.assertTrue(any('.head.' in name for name in changed))

    def test_eval_and_state_roundtrip_are_deterministic(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.images, self.images)
            torch.testing.assert_close(expected, self.model(self.images, self.images), rtol=0, atol=0)
        restored = MDPIFusionModel(TinyDualTrack()).eval()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(self.images, self.images), rtol=0, atol=0)
        report = restored.scope_report()
        self.assertEqual(report['trainable_parameters'], sum(p.numel() for p in restored.parameters() if p.requires_grad))
        self.assertTrue(report['frozen_encoders_in_eval_mode'])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.images[:, :9], self.images)


class CheckpointBuilderTests(unittest.TestCase):
    def test_loads_all_original_weights_before_reducing_head(self):
        original = TinyDualTrack()
        checkpoint = {
            'model': copy.deepcopy(original.state_dict()),
            'config': {'model': {
                'name': 'dualtrack_fusion_model',
                'checkpoint': '/obsolete/stage1.pt',
                'global_encoder_cfg': {'name': 'global_encoder_cnn'},
                'local_encoder_cfg': {'name': 'dualtrack_loc_enc_stg3_legacy'},
            }},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.pt'
            torch.save(checkpoint, path)
            with patch('src.models.get_model', return_value=TinyDualTrack()) as factory:
                model = build_mdpi_fusion(path)
            called = factory.call_args.kwargs
            self.assertNotIn('checkpoint', called)
            self.assertEqual(called['global_encoder_cfg']['position_embedding_type'], 'absolute')
            self.assertEqual(called['grid_spacing'], 2)
            torch.testing.assert_close(model.dualtrack.head.fc.weight, original.head.fc.weight[[1, 2]])
            torch.testing.assert_close(model.dualtrack.fusion_module.fc.weight, original.fusion_module.fc.weight)
            checkpoint['model'].pop('fusion_module.fc.bias')
            torch.save(checkpoint, path)
            with patch('src.models.get_model', return_value=TinyDualTrack()):
                with self.assertRaises(RuntimeError):
                    build_mdpi_fusion(path)


if __name__ == '__main__':
    unittest.main()
