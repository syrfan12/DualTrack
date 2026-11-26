"""Training and checkpoint IO for original-image MDPI fusion adaptation."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.mdpi_original import MDPIOriginalDataset, file_sha256
from src.models.mdpi_fusion import (
    MODEL_SCHEMA, build_mdpi_fusion, pair_translations, targets_to_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_SCHEMA = "mdpi_original_fusion_run_v1"
SELECTION_METRIC = "mean_pair_error_nominal_mm"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, config, train=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=train,
        num_workers=config["num_workers"], pin_memory=config["device"].startswith("cuda"),
        worker_init_fn=seed_worker, generator=generator,
        # Recreate workers each epoch so an interrupted/resumed iterator draws
        # the same worker-base seed and shuffle order as an uninterrupted run.
        persistent_workers=False,
    )


def autocast_context(device, enabled):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def batch_loss(model, batch, device, use_amp=False):
    with autocast_context(device, use_amp):
        steps = model(batch["global_encoder_images"].to(device),
                      batch["local_encoder_images"].to(device))
    # Integrate and calculate loss in float32 even when attention uses AMP.
    predicted = pair_translations(steps.float())
    target = targets_to_pairs(batch["positions_yz"].to(device).float())
    loss = torch.nn.functional.mse_loss(predicted, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite MDPI pair loss")
    return loss, predicted, target


def code_hashes():
    paths = [ROOT / "src/mdpi_training.py", ROOT / "src/mdpi_evaluation.py",
             ROOT / "src/datasets/mdpi_original.py"]
    paths.extend(sorted((ROOT / "src/models").rglob("*.py")))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": {"name": state[0], "values": state[1].tolist(), "position": state[2],
                  "has_gauss": state[3], "cached_gaussian": state[4]},
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["values"], np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def checkpoint_bundle(model, config, dataset, baselines, provenance, epoch,
                      score, best_score, stale_epochs, optimizer, scheduler, scaler,
                      loader_generator):
    return {
        "schema": RUN_SCHEMA, "model_schema": MODEL_SCHEMA,
        "config": copy.deepcopy(config), "manifest_sha256": dataset.manifest_sha256,
        "pretrained_sha256": provenance["pretrained_sha256"],
        "code_sha256": provenance["code_sha256"], "baselines": baselines,
        "units": "nominal_mm_against_interpolated_reference",
        "coordinate_convention": "ty=-reference_Y; tz=reference_X; tx=0; R=I",
        "selection_metric": SELECTION_METRIC, "epoch": epoch, "score": score,
        "best_score": best_score, "stale_epochs": stale_epochs,
        "model": model.state_dict(), "scope": model.scope_report(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "rng": rng_state(),
        "loader_generator": loader_generator.get_state(),
    }


def save_checkpoint(path, bundle):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)


def load_run_checkpoint(checkpoint_path, manifest_path=None, device="cpu", verify_sources=True):
    """Strictly reconstruct the model and verify its input/provenance contract.

    Source-byte checking belongs to MDPIOriginalDataset and is performed by the
    caller for the requested split; this function checks the manifest digest.
    """
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (not isinstance(bundle, dict) or bundle.get("schema") != RUN_SCHEMA
            or bundle.get("model_schema") != MODEL_SCHEMA
            or bundle.get("selection_metric") != SELECTION_METRIC):
        raise ValueError("Incompatible MDPI fusion checkpoint schema")
    manifest_path = Path(manifest_path or bundle["config"]["manifest"]).resolve()
    if file_sha256(manifest_path) != bundle["manifest_sha256"]:
        raise ValueError("Dataset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "mdpi_original_dualtrack_v1":
        raise ValueError("Incompatible original-image dataset schema")
    pretrained = Path(bundle["config"]["pretrained"])
    if file_sha256(pretrained) != bundle["pretrained_sha256"]:
        raise ValueError("Original pretrained checkpoint hash mismatch")
    if verify_sources:
        for name, digest in bundle["code_sha256"].items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError(f"Model/training code hash mismatch: {name}")
    model = build_mdpi_fusion(pretrained, grid_spacing=bundle["config"]["grid_spacing"], device=device)
    model.load_state_dict(bundle["model"], strict=True)
    model.eval()
    return model, bundle, manifest_path


def _tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(model):
    encoders = (("local", model.dualtrack.local_encoder),
                ("global", model.dualtrack.global_encoder))
    return {name: _tensor_digest(module.state_dict().items()) for name, module in encoders}


def make_optimizer(model, config):
    groups = model.trainable_parameter_groups(config["fusion_lr"], config["head_lr"])
    parameters = [p for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(parameters) != len({id(p) for p in parameters}) or {id(p) for p in parameters} != expected:
        raise RuntimeError("Optimizer groups do not exactly match the trainable scope")
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def scaled_optimizer_step(loss, model, optimizer, scaler, max_norm):
    """Let GradScaler skip expected FP16 overflow while adapting its scale."""
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm,
        error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return {"updated": scaler.get_scale() >= previous_scale,
            "gradient_norm": float(norm) if torch.isfinite(norm) else None}


def check_weights(model, dataset, config, optimize=False):
    """One real train window; optimizer-step verification is explicitly opt-in."""
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    model.train(optimize)
    before = frozen_digest(model)
    fusion_before = _tensor_digest(model.dualtrack.fusion_module.state_dict().items())
    head_before = _tensor_digest(model.dualtrack.head.state_dict().items())
    context = contextlib.nullcontext() if optimize else torch.no_grad()
    with context:
        loss, predicted, _ = batch_loss(model, batch, config["device"], config["amp"])
    if optimize:
        optimizer = make_optimizer(model, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise RuntimeError(f"Missing/non-finite trainable gradient: {name}")
            if not p.requires_grad and p.grad is not None:
                raise RuntimeError(f"Unexpected frozen gradient: {name}")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      config["grad_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
    unchanged = before == frozen_digest(model)
    if not unchanged:
        raise RuntimeError("Frozen encoder parameters or buffers changed")
    fusion_changed = fusion_before != _tensor_digest(model.dualtrack.fusion_module.state_dict().items())
    head_changed = head_before != _tensor_digest(model.dualtrack.head.state_dict().items())
    if optimize and not (fusion_changed and head_changed):
        raise RuntimeError("Fusion and head must both update in the smoke step")
    return {"mode": "one_optimizer_step" if optimize else "forward_only",
            "pair_output_shape": list(predicted.shape), "mse_yz": float(loss.detach()),
            "encoders_and_buffers_unchanged": unchanged,
            "fusion_changed": fusion_changed, "head_changed": head_changed,
            "scope": model.scope_report(), "manifest_sha256": dataset.manifest_sha256}


def train(config, resume=None, skip_trajectory_eval=False):
    from src.mdpi_evaluation import evaluate_windows, evaluate_trajectories, fit_train_baselines

    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Choose a new run output directory: {output}")
    seed_everything(config["seed"])
    train_data = MDPIOriginalDataset(config["manifest"], "train")
    val_data = MDPIOriginalDataset(config["manifest"], "val")
    baselines = fit_train_baselines(train_data)
    if resume:
        model, old, _ = load_run_checkpoint(resume, config["manifest"], config["device"])
        mutable = {"output", "device", "num_workers", "cpu_threads"}
        if any(old["config"].get(k) != v for k, v in config.items() if k not in mutable):
            raise ValueError("Resume configuration differs; retain the original training settings")
        if old["epoch"] >= config["max_epochs"] or old["stale_epochs"] >= config["patience"]:
            raise ValueError("This run has already reached its stopping condition")
    else:
        model = build_mdpi_fusion(config["pretrained"], config["grid_spacing"], config["device"])
        old = None
    optimizer = make_optimizer(model, config)
    warmup = config["warmup_epochs"]

    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, config["max_epochs"] - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and config["device"].startswith("cuda"))
    train_loader = make_loader(train_data, config, train=True)
    val_loader = make_loader(val_data, config)
    provenance = {"pretrained_sha256": file_sha256(config["pretrained"]), "code_sha256": code_hashes()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", dict(provenance, manifest_sha256=train_data.manifest_sha256,
                                               scope=model.scope_report(), baselines=baselines))
    before = frozen_digest(model)
    initial = evaluate_windows(model, val_loader, config["device"], config["amp"], baselines)
    write_json(output / "epoch0_validation.json", initial)
    start_epoch, stale = 1, 0
    best_score = initial[SELECTION_METRIC]
    if old:
        optimizer.load_state_dict(old["optimizer"])
        scheduler.load_state_dict(old["scheduler"])
        scaler.load_state_dict(old["scaler"])
        start_epoch, stale = old["epoch"] + 1, old["stale_epochs"]
        best_score = old["best_score"]
        # Recover the previous best, not necessarily the checkpoint being resumed.
        previous_best = Path(resume).parent / "best.pt"
        if not previous_best.is_file():
            raise FileNotFoundError("Resume requires best.pt next to the resume checkpoint")
        _, previous, _ = load_run_checkpoint(previous_best, config["manifest"], config["device"])
        if previous["score"] != best_score:
            raise ValueError("Resume best checkpoint score mismatch")
        save_checkpoint(output / "checkpoints/best.pt", previous)
        restore_rng(old["rng"])
        train_loader.generator.set_state(old["loader_generator"].cpu())
    else:
        bundle = checkpoint_bundle(model, config, train_data, baselines, provenance, 0, best_score,
                                   best_score, stale, optimizer, scheduler, scaler, train_loader.generator)
        save_checkpoint(output / "checkpoints/best.pt", bundle)
    patience_reference = old.get("patience_reference", best_score) if old else best_score
    print(f"epoch=0 val_pair_error={initial[SELECTION_METRIC]:.6f} nominal_mm", flush=True)
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        model.train()
        sums, count, skipped_steps = 0.0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = batch_loss(model, batch, config["device"], config["amp"])
            step = scaled_optimizer_step(loss, model, optimizer, scaler, config["grad_clip_norm"])
            skipped_steps += int(not step["updated"])
            batch_count = len(batch["positions_yz"])
            sums += float(loss.detach()) * batch_count
            count += batch_count
        if before != frozen_digest(model):
            raise RuntimeError("Frozen encoder parameters or buffers changed during training")
        if skipped_steps == len(train_loader):
            raise FloatingPointError("AMP skipped every optimizer step; reduce the loss scale or disable AMP")
        validation = evaluate_windows(model, val_loader, config["device"], config["amp"], baselines)
        score = validation[SELECTION_METRIC]
        improved = score < best_score
        if improved:
            best_score = score
        if score < patience_reference - config["min_delta"]:
            patience_reference, stale = score, 0
        else:
            stale += 1
        scheduler.step()
        row = {"epoch": epoch, "train_mse_yz": sums / count, "validation": validation,
               "best_score": best_score, "stale_epochs": stale,
               "amp_skipped_optimizer_steps": skipped_steps,
               "learning_rates": [g["lr"] for g in optimizer.param_groups]}
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        bundle = checkpoint_bundle(model, config, train_data, baselines, provenance, epoch, score,
                                   best_score, stale, optimizer, scheduler, scaler, train_loader.generator)
        # Keeping the patience reference makes an interrupted run resume exactly.
        bundle["patience_reference"] = patience_reference
        if improved:
            save_checkpoint(output / "checkpoints/best.pt", bundle)
        save_checkpoint(output / "checkpoints/last.pt", bundle)
        print(f"epoch={epoch} train_mse_yz={sums/count:.6f} val_pair_error={score:.6f} "
              f"best={best_score:.6f} patience={stale}/{config['patience']}", flush=True)
        if stale >= config["patience"]:
            break
    write_json(output / "best_validation.json", {"metric": SELECTION_METRIC, "value": best_score})
    if not skip_trajectory_eval:
        best_model, best_bundle, _ = load_run_checkpoint(output / "checkpoints/best.pt",
                                                       config["manifest"], config["device"])
        evaluate_trajectories(best_model, val_data, output / "trajectory_validation_best",
                              batch_size=config["batch_size"], device=config["device"],
                              baselines=best_bundle["baselines"])
    return {"output": str(output), "best_validation_pair_error_nominal_mm": best_score}
