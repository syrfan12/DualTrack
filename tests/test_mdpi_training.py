"""Synthetic end-to-end checks for MDPI checkpoint and training state contracts."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import src.mdpi_training as training
from src.datasets.mdpi_original import file_sha256
from src.models.mdpi_fusion import MDPIFusionModel, MODEL_SCHEMA


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(1, 4)
        self.bn = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.8)
        self.projection = nn.Linear(4, 4)

    def forward(self, images, indices=None):
        batch, length = images.shape[:2]
        x = self.image(images.mean(dim=(-1, -2, -3)).unsqueeze(-1))
        x = self.bn(x.reshape(-1, 4)).reshape(batch, length, 4)
        return self.projection(self.dropout(x))


class TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.35)

    def forward(self, local, encoder_hidden_states):
        return self.fc(self.dropout(local + encoder_hidden_states.mean(dim=1, keepdim=True)))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)

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


def tiny_model_factory(_checkpoint=None, grid_spacing=2, device='cpu'):
    return MDPIFusionModel(TinyDualTrack()).to(device)


class TinyOriginalDataset(Dataset):
    """No MDPI files or image bytes are opened by these trainer integration tests."""

    def __init__(self, manifest_path, split='train', **_kwargs):
        if split not in ('train', 'val'):
            raise ValueError('Synthetic integration fixture only permits train/val')
        self.split = split
        self.manifest_path = Path(manifest_path)
        self.manifest_sha256 = file_sha256(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.positions = torch.stack((torch.arange(30) * 0.07, torch.arange(30).square() * 0.003), dim=-1)
        self.scans = {split: {'split': split, 'positions_yz': self.positions.tolist()}}
        self.windows = [{'scan_id': split, 'frame_indices': list(range(i, i + 10))} for i in (0, 10, 20)]
        generator = torch.Generator().manual_seed(123 if split == 'train' else 456)
        self.images = torch.rand((30, 1, 4, 4), generator=generator)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        indices = self.windows[index]['frame_indices']
        return {'local_encoder_images': self.images[indices],
                'global_encoder_images': self.images[indices],
                'positions_yz': self.positions[indices],
                'scan_id': self.split, 'source_indices': torch.tensor(indices),
                'window_id': index}


class InterruptedRun(RuntimeError):
    pass


class CPUGradScaler:
    """CPU scaler double preserving unscale-time overflow detection semantics."""

    def __init__(self, overflow=False):
        self.current_scale = 8.0
        self.overflow = overflow
        self.found_inf = False
        self.optimizer_updates = 0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.current_scale

    def scale(self, loss):
        return loss * self.current_scale

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self.current_scale)
                    if self.overflow:
                        parameter.grad.fill_(float('inf'))
                    self.found_inf |= not bool(torch.isfinite(parameter.grad).all())

    def step(self, optimizer):
        # Real GradScaler records found_inf before clipping can turn inf to NaN.
        if not self.found_inf:
            optimizer.step()
            self.optimizer_updates += 1

    def update(self):
        if self.found_inf:
            self.current_scale *= 0.5


class ScaledOptimizerStepTests(unittest.TestCase):
    def _fixture(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = model(torch.ones(1, 1)).square().mean()
        return model, optimizer, loss

    def test_amp_overflow_skips_optimizer_update_and_reduces_scale(self):
        model, optimizer, loss = self._fixture()
        before = model.weight.detach().clone()
        scaler = CPUGradScaler(overflow=True)
        result = training.scaled_optimizer_step(loss, model, optimizer, scaler, 1.)
        self.assertFalse(result['updated'])
        self.assertIsNone(result['gradient_norm'])
        self.assertEqual(scaler.optimizer_updates, 0)
        self.assertEqual(scaler.get_scale(), 4.)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def test_finite_scaled_step_unscales_clips_and_updates_weights(self):
        model, optimizer, loss = self._fixture()
        scaler = CPUGradScaler()
        result = training.scaled_optimizer_step(loss, model, optimizer, scaler, 1.)
        self.assertTrue(result['updated'])
        self.assertEqual(scaler.optimizer_updates, 1)
        self.assertEqual(scaler.get_scale(), 8.)
        self.assertAlmostEqual(result['gradient_norm'], 2., places=5)
        # Gradient 2 is unscaled then clipped to 1, giving an SGD update of 0.1.
        self.assertAlmostEqual(float(model.weight.detach()), 0.9, places=5)

    def test_disabled_scaler_nonfinite_gradient_raises_before_optimizer_update(self):
        model, optimizer, loss = self._fixture()
        before = model.weight.detach().clone()
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            training.scaled_optimizer_step(loss, model, optimizer, scaler, 1.)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({'schema': 'mdpi_original_dualtrack_v1'}))
        self.pretrained = self.root / 'original.pt'
        self.pretrained.write_bytes(b'synthetic-original-checkpoint-hash-fixture')
        self.config = {
            'manifest': str(self.manifest), 'pretrained': str(self.pretrained),
            'output': str(self.root / 'run'), 'device': 'cpu', 'num_workers': 0,
            'cpu_threads': 1, 'batch_size': 2, 'seed': 71, 'grid_spacing': 2,
            'fusion_lr': 0.005, 'head_lr': 0.01, 'weight_decay': 0.0001,
            'warmup_epochs': 1, 'max_epochs': 3, 'patience': 8,
            'min_delta': 0.0001, 'grad_clip_norm': 1.0, 'amp': False,
        }
        self.stack = []
        for name, replacement in (
            ('src.mdpi_training.MDPIOriginalDataset', TinyOriginalDataset),
            ('src.mdpi_training.build_mdpi_fusion', tiny_model_factory),
            ('src.mdpi_training.code_hashes', lambda: {}),
        ):
            patcher = patch(name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _bundle(self):
        model = tiny_model_factory()
        dataset = TinyOriginalDataset(self.manifest)
        optimizer = training.make_optimizer(model, self.config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        loader_generator = torch.Generator().manual_seed(9)
        bundle = training.checkpoint_bundle(
            model, self.config, dataset, {'fit_scope': 'train_only'},
            {'pretrained_sha256': file_sha256(self.pretrained), 'code_sha256': {}},
            1, 0.25, 0.2, 2, optimizer, scheduler, scaler, loader_generator,
        )
        return model, bundle

    def _save(self, bundle, name='bundle.pt'):
        path = self.root / name
        training.save_checkpoint(path, bundle)
        return path

    def _load(self, path):
        return training.load_run_checkpoint(path, self.manifest, 'cpu')

    def test_rng_state_roundtrip_reproduces_python_numpy_and_torch(self):
        training.seed_everything(81)
        state = training.rng_state()
        expected = (random.random(), np.random.rand(5), torch.rand(5))
        training.seed_everything(3)
        training.restore_rng(state)
        actual = (random.random(), np.random.rand(5), torch.rand(5))
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)

    def test_checkpoint_strict_roundtrip_preserves_output_and_training_metadata(self):
        model, bundle = self._bundle()
        model.eval()
        images = torch.rand(1, 10, 1, 4, 4)
        with torch.no_grad():
            expected = model(images, images)
        restored, loaded, manifest = self._load(self._save(bundle))
        with torch.no_grad():
            torch.testing.assert_close(restored(images, images), expected, rtol=0, atol=0)
        self.assertEqual(loaded['epoch'], 1)
        self.assertEqual(loaded['stale_epochs'], 2)
        self.assertEqual(loaded['best_score'], 0.2)
        self.assertEqual(manifest, self.manifest)
        self.assertFalse(restored.training)
        self.assertEqual(bundle['scope'], restored.scope_report())

    def test_schema_selection_and_manifest_hash_fail_before_model_creation(self):
        _, bundle = self._bundle()
        cases = [('schema', 'old-run'), ('model_schema', 'efficientnet-chain'),
                 ('selection_metric', 'different-selection-metric'),
                 ('manifest_sha256', '0' * 64)]
        for key, bad_value in cases:
            wrong = copy.deepcopy(bundle)
            wrong[key] = bad_value
            path = self._save(wrong, key + '.pt')
            with self.subTest(key=key), patch('src.mdpi_training.build_mdpi_fusion') as builder:
                with self.assertRaises(ValueError):
                    self._load(path)
                builder.assert_not_called()
        self.manifest.write_text(json.dumps({'schema': 'wrong-dataset'}))
        bundle['manifest_sha256'] = file_sha256(self.manifest)
        with patch('src.mdpi_training.build_mdpi_fusion') as builder:
            with self.assertRaisesRegex(ValueError, 'dataset schema'):
                self._load(self._save(bundle))
            builder.assert_not_called()

    def test_pretrained_and_code_hash_mismatches_fail_before_model_creation(self):
        _, bundle = self._bundle()
        for field, value in (('pretrained_sha256', '0' * 64),
                             ('code_sha256', {'src/mdpi_training.py': '0' * 64})):
            wrong = copy.deepcopy(bundle)
            wrong[field] = value
            with self.subTest(field=field), patch('src.mdpi_training.build_mdpi_fusion') as builder:
                with self.assertRaises(ValueError):
                    self._load(self._save(wrong, field + '.pt'))
                builder.assert_not_called()

    def test_state_dictionary_missing_or_extra_entries_are_rejected(self):
        _, bundle = self._bundle()
        for extra in (False, True):
            wrong = copy.deepcopy(bundle)
            if extra:
                wrong['model']['unexpected.weight'] = torch.zeros(1)
            else:
                wrong['model'].pop('dualtrack.head.fc.bias')
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                self._load(self._save(wrong, 'bad-state.pt'))

    def test_two_epoch_train_preserves_all_frozen_tensors_and_writes_reloadable_state(self):
        config = dict(self.config, max_epochs=2)
        training.seed_everything(config['seed'])
        original = tiny_model_factory()
        frozen_before = training.frozen_digest(original)
        training.train(config, skip_trajectory_eval=True)
        output = Path(config['output'])
        last_model, last, _ = self._load(output / 'checkpoints/last.pt')
        self.assertEqual(last['epoch'], 2)
        self.assertEqual(training.frozen_digest(last_model), frozen_before)
        for prefix in ('dualtrack.fusion_module.', 'dualtrack.head.'):
            self.assertTrue(any(not torch.equal(value, original.state_dict()[name])
                                for name, value in last_model.state_dict().items() if name.startswith(prefix)))
        self.assertTrue(last['optimizer']['state'])
        self.assertIn('rng', last)
        self.assertIn('loader_generator', last)
        self.assertEqual(len((output / 'history.jsonl').read_text().splitlines()), 2)
        best, best_bundle, _ = self._load(output / 'checkpoints/best.pt')
        self.assertEqual(best_bundle['model_schema'], MODEL_SCHEMA)
        self.assertTrue(best.scope_report()['frozen_encoders_in_eval_mode'])
        with self.assertRaises(FileExistsError):
            training.train(config, skip_trajectory_eval=True)

    def test_patience_reference_tracks_cumulative_significant_improvement(self):
        config = dict(self.config, max_epochs=5, patience=2, min_delta=0.01)
        scores = [1.0, 0.995, 0.989, 0.988, 0.987]
        with patch('src.mdpi_evaluation.evaluate_windows',
                   side_effect=[{training.SELECTION_METRIC: score} for score in scores]):
            training.train(config, skip_trajectory_eval=True)
        checkpoint = torch.load(Path(config['output']) / 'checkpoints/last.pt', weights_only=True)
        self.assertEqual(checkpoint['epoch'], 4)
        self.assertEqual(checkpoint['stale_epochs'], 2)
        self.assertEqual(checkpoint['patience_reference'], 0.989)
        self.assertEqual(checkpoint['best_score'], 0.987)
        stopped = dict(config, output=str(self.root / 'stopped-resume'))
        with self.assertRaisesRegex(ValueError, 'stopping condition'):
            training.train(stopped, resume=Path(config['output']) / 'checkpoints/last.pt',
                           skip_trajectory_eval=True)
        self.assertFalse(Path(stopped['output']).exists())

    def test_epoch_with_every_optimizer_step_skipped_is_rejected(self):
        with patch('src.mdpi_training.scaled_optimizer_step',
                   return_value={'updated': False, 'gradient_norm': None}):
            with self.assertRaisesRegex(FloatingPointError, 'every optimizer step'):
                training.train(dict(self.config, max_epochs=1), skip_trajectory_eval=True)
        self.assertFalse((Path(self.config['output']) / 'checkpoints/last.pt').exists())

    def test_partial_skipped_optimizer_steps_are_logged(self):
        actual_step = training.scaled_optimizer_step
        calls = []

        def skip_first_step(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                return {'updated': False, 'gradient_norm': None}
            return actual_step(*args, **kwargs)

        with patch('src.mdpi_training.scaled_optimizer_step', side_effect=skip_first_step):
            training.train(dict(self.config, max_epochs=1), skip_trajectory_eval=True)
        self.assertEqual(len(calls), 2)
        history = Path(self.config['output']) / 'history.jsonl'
        row = json.loads(history.read_text().splitlines()[0])
        self.assertEqual(row['amp_skipped_optimizer_steps'], 1)

    def _assert_resume_matches_uninterrupted(self, num_workers):
        config = dict(self.config, num_workers=num_workers, output=str(self.root / 'continuous'))
        training.train(config, skip_trajectory_eval=True)
        interrupted = dict(config, output=str(self.root / 'interrupted'))
        save = training.save_checkpoint

        def save_then_interrupt(path, bundle):
            save(path, bundle)
            if Path(path).name == 'last.pt' and bundle['epoch'] == 1:
                raise InterruptedRun('Synthetic interruption after complete checkpoint save')

        with patch('src.mdpi_training.save_checkpoint', side_effect=save_then_interrupt):
            with self.assertRaises(InterruptedRun):
                training.train(interrupted, skip_trajectory_eval=True)
        resumed = dict(config, output=str(self.root / 'resumed'))
        training.train(resumed, resume=Path(interrupted['output']) / 'checkpoints/last.pt',
                       skip_trajectory_eval=True)
        full = torch.load(Path(config['output']) / 'checkpoints/last.pt', weights_only=True)
        restored = torch.load(Path(resumed['output']) / 'checkpoints/last.pt', weights_only=True)
        self.assertEqual(full['epoch'], restored['epoch'])
        self.assertEqual(full['stale_epochs'], restored['stale_epochs'])
        self.assertEqual(full['best_score'], restored['best_score'])
        self.assertEqual(full['patience_reference'], restored['patience_reference'])
        self.assertEqual(full['scheduler'], restored['scheduler'])
        for name in full['model']:
            torch.testing.assert_close(full['model'][name], restored['model'][name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(full['rng']['torch'], restored['rng']['torch'], rtol=0, atol=0)
        torch.testing.assert_close(full['loader_generator'], restored['loader_generator'], rtol=0, atol=0)

    def test_resume_reproduces_uninterrupted_rng_model_and_scheduler(self):
        self._assert_resume_matches_uninterrupted(num_workers=0)

    def test_resume_reproduces_worker_loader_order(self):
        self._assert_resume_matches_uninterrupted(num_workers=1)


if __name__ == '__main__':
    unittest.main()
