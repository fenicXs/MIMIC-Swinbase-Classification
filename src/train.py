from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpointing import load_checkpoint, save_checkpoint
from .data_utils import compute_pos_weight, load_class_names, load_config, load_split_csv
from .dataset import MIMICCXRJPGDataset
from .ema_utils import EMAWrapper
from .metrics import evaluate_auc
from .model import create_swin_base
from .plotting import plot_train_loss, plot_val_mean_auc

def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str = "module.") -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(isinstance(k, str) and k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _get_ema_state_dict(ema_wrapper: Optional[EMAWrapper]) -> Optional[Dict[str, Any]]:
    if ema_wrapper is None:
        return None
    ema_obj = ema_wrapper.ema
    # timm ModelEmaV2 typically keeps weights in ema_obj.module
    if hasattr(ema_obj, "module"):
        try:
            return ema_obj.module.state_dict()
        except Exception:
            pass
    try:
        return ema_obj.state_dict()
    except Exception:
        return None


def _load_ema_state_dict(ema_wrapper: Optional[EMAWrapper], state: Dict[str, Any]) -> bool:
    if ema_wrapper is None or not state:
        return False
    ema_obj = ema_wrapper.ema
    candidates = []
    candidates.append(ema_obj)
    if hasattr(ema_obj, "module"):
        candidates.append(ema_obj.module)

    # Try strict load first, then with stripped prefix
    for sd in (state, _strip_prefix_if_present(state)):
        for target in candidates:
            try:
                target.load_state_dict(sd, strict=True)
                return True
            except Exception:
                continue
    return False


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed > determinism for training
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Swin-Base on MIMIC-CXR-JPG (PA-only) with EMA.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from. If not set and auto_resume is true, "
             "will resume from checkpoints/<exp_name>/last.pth if it exists.",
    )
    return p.parse_args()


def save_history_csv(history: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_path, index=False)


def maybe_load_history_csv(history_path: Path) -> Optional[Dict[str, List[float]]]:
    if not history_path.exists():
        return None
    df = pd.read_csv(history_path)
    hist: Dict[str, List[float]] = {}
    for c in df.columns:
        hist[c] = df[c].tolist()
    return hist


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = cfg.get("experiment_name", "swin_base_mimic")
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    mimic_root = Path(data_cfg["mimic_root"])
    train_csv = data_cfg["train_csv"]
    val_csv = data_cfg["val_csv"]
    test_csv = data_cfg.get("test_csv", None)
    class_names_json = data_cfg["class_names_json"]

    image_size = int(data_cfg.get("image_size", 512))
    num_workers = int(data_cfg.get("num_workers", 8))

    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 50))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    mixed_precision = bool(train_cfg.get("mixed_precision", True))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    eval_every = int(train_cfg.get("eval_every", 1))

    ema_cfg = train_cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", True))
    ema_decay = float(ema_cfg.get("decay", 0.999))

    checkpoint_root = Path(train_cfg.get("checkpoint_dir", "./checkpoints"))
    plots_root = Path(train_cfg.get("plots_dir", "./plots"))

    ckpt_dir = checkpoint_root / exp_name
    plot_dir = plots_root / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    history_path = ckpt_dir / "history.csv"
    train_loss_plot_path = plot_dir / "train_loss.png"
    val_auc_plot_path = plot_dir / "val_mean_auc.png"

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", True))
    es_patience = int(es_cfg.get("patience", 7))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))

    top_k = int(train_cfg.get("top_k", 3))

    # Load class names (CheXpert 14)
    class_names = load_class_names(class_names_json)
    num_classes = len(class_names)
    print(f"[train] num_classes={num_classes}")
    print(f"[train] class_names={class_names}")

    # Load prepared splits
    train_df = load_split_csv(train_csv, class_names)
    val_df = load_split_csv(val_csv, class_names)
    print(f"[train] train rows={len(train_df):,}, val rows={len(val_df):,}")

    # pos_weight from training set
    pos_weight_np = compute_pos_weight(train_df, class_names)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print("[train] pos_weight:", pos_weight_np)

    # Datasets
    train_dataset = MIMICCXRJPGDataset(
        df=train_df,
        mimic_root=mimic_root,
        class_names=class_names,
        image_size=image_size,
        train=True,
    )
    val_dataset = MIMICCXRJPGDataset(
        df=val_df,
        mimic_root=mimic_root,
        class_names=class_names,
        image_size=image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = create_swin_base(num_classes=num_classes, pretrained=True, img_size=image_size)
    model = model.to(device)

    # Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - max(0, warmup_epochs)))

    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and device.type == "cuda")

    # EMA
    ema_wrapper: Optional[EMAWrapper] = None
    if ema_enabled:
        print(f"[train] EMA enabled (decay={ema_decay})")
        ema_wrapper = EMAWrapper(model, decay=ema_decay, device=None)

    # Resume logic
    auto_resume = bool(train_cfg.get("auto_resume", True))
    resume_path = args.resume or train_cfg.get("resume", None)
    last_path = ckpt_dir / "last.pth"
    if resume_path is None and auto_resume and last_path.exists():
        resume_path = str(last_path)

    start_epoch = 0
    best_val_auc = float("-inf")
    best_epoch = -1
    es_counter = 0
    best_ckpts: List[Dict[str, Any]] = []
    history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_mean_auc": []}

    if resume_path is not None:
        print(f"[train] Resuming from checkpoint: {resume_path}")
        ckpt = load_checkpoint(resume_path, device=device)

        # epoch stored as "next epoch to run"
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_auc = float(ckpt.get("best_val_auc", best_val_auc))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        es_counter = int(ckpt.get("es_counter", es_counter))
        best_ckpts = ckpt.get("best_ckpts", best_ckpts) or best_ckpts

        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            print(f"[train] WARNING: could not load scheduler_state ({e}); using fresh scheduler.")
        if "scaler_state" in ckpt and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"[train] WARNING: could not load scaler_state ({e}).")
        if ema_wrapper is not None and "ema_state" in ckpt:
            try:
                _load_ema_state_dict(ema_wrapper, ckpt["ema_state"])
            except Exception as e:
                print(f"[train] WARNING: could not load ema_state ({e}).")

        # History
        if "history" in ckpt and isinstance(ckpt["history"], dict):
            history = ckpt["history"]
        else:
            hist_csv = maybe_load_history_csv(history_path)
            if hist_csv is not None:
                history = hist_csv

        print(f"[train] start_epoch={start_epoch}, best_val_auc={best_val_auc:.4f}, best_epoch={best_epoch}, es_counter={es_counter}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        # Warmup (simple linear warmup)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_factor

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=110)
        optimizer.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
                loss = loss / max(1, grad_accum_steps)

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                if max_grad_norm is not None and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if ema_wrapper is not None:
                    ema_wrapper.update(model)

            running_loss += loss.item() * images.size(0) * max(1, grad_accum_steps)
            pbar.set_postfix(loss=float(loss.item() * max(1, grad_accum_steps)))

        # If using gradient accumulation, we may have leftover grads if total steps is not divisible by grad_accum_steps
        if grad_accum_steps > 1:
            last_step = len(train_loader)
            if last_step % grad_accum_steps != 0:
                if max_grad_norm is not None and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema_wrapper is not None:
                    ema_wrapper.update(model)

        # Step scheduler after warmup
        if warmup_epochs <= 0 or epoch >= warmup_epochs:
            scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        do_eval = ((epoch + 1) % eval_every == 0)
        if do_eval:
            eval_model = ema_wrapper.ema if ema_wrapper is not None else model
            val_mean_auc, _ = evaluate_auc(
                eval_model,
                val_loader,
                device=device,
                class_names=class_names,
                use_mixed_precision=mixed_precision,
            )
        else:
            val_mean_auc = float("nan")

        print(f"[train] Epoch {epoch+1}/{epochs} | train_loss={train_loss:.6f} | val_mAUC={val_mean_auc}")

        # History
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["val_mean_auc"].append(float(val_mean_auc))
        save_history_csv(history, history_path)

        # Plot (overwrite same images each epoch)
        try:
            plot_train_loss(history, train_loss_plot_path, title=f"{exp_name} - Train Loss")
            plot_val_mean_auc(history, val_auc_plot_path, title=f"{exp_name} - Val Mean AUC")
        except Exception as e:
            print(f"[train] WARNING: plotting failed: {e}")

        # Checkpoint bookkeeping
        improved = False
        if do_eval and not (np.isnan(val_mean_auc)):
            if val_mean_auc > (best_val_auc + es_min_delta):
                improved = True
                best_val_auc = float(val_mean_auc)
                best_epoch = epoch + 1
                es_counter = 0
            else:
                es_counter += 1

        # Save BEST (EMA weights included in checkpoint so evaluate.py can load EMA)
        if improved:
            best_state = {
                "epoch": epoch + 1,  # next epoch will be epoch+1 (0-index) but store human-readable? keep both
                "epoch_next": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "ema_state": _get_ema_state_dict(ema_wrapper),
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
                "es_counter": es_counter,
                "class_names": class_names,
                "history": history,
                "config_path": str(args.config),
            }
            save_checkpoint(best_state, ckpt_dir / "best.pth")
            print(f"[train] ✅ New best checkpoint saved: best.pth (val_mAUC={best_val_auc:.6f} @ epoch {best_epoch})")

        # Top-k optional (by val mAUC)
        if do_eval and not (np.isnan(val_mean_auc)):
            ckpt_filename = f"epoch{epoch+1:03d}_val{val_mean_auc:.4f}.pth"
            ckpt_path = ckpt_dir / ckpt_filename
            candidate = {"epoch": epoch + 1, "val_mean_auc": float(val_mean_auc), "path": ckpt_filename}

            # Save only if in top-k
            if len(best_ckpts) < top_k or val_mean_auc > min(x["val_mean_auc"] for x in best_ckpts):
                state = {
                    "epoch": epoch + 1,
                    "epoch_next": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "ema_state": _get_ema_state_dict(ema_wrapper),
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                    "es_counter": es_counter,
                    "class_names": class_names,
                    "history": history,
                    "config_path": str(args.config),
                }
                save_checkpoint(state, ckpt_path)

                best_ckpts.append(candidate)
                best_ckpts = sorted(best_ckpts, key=lambda x: x["val_mean_auc"], reverse=True)

                # prune extras
                if len(best_ckpts) > top_k:
                    extras = best_ckpts[top_k:]
                    best_ckpts = best_ckpts[:top_k]
                    for ex in extras:
                        p = ckpt_dir / ex["path"]
                        if p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass

                with open(ckpt_dir / "top_k.json", "w", encoding="utf-8") as f:
                    json.dump({"top_k": top_k, "checkpoints": best_ckpts}, f, indent=2)

        # ALWAYS save LAST (overwritten)
        last_state = {
            # store epoch as the next epoch to run (0-index friendly)
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "ema_state": _get_ema_state_dict(ema_wrapper),
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "es_counter": es_counter,
            "best_ckpts": best_ckpts,
            "class_names": class_names,
            "history": history,
            "config_path": str(args.config),
        }
        save_checkpoint(last_state, last_path)

        # Early stopping
        if es_enabled and do_eval and not (np.isnan(val_mean_auc)):
            if es_counter >= es_patience:
                print(
                    f"[train] 🛑 Early stopping triggered: no improvement for {es_patience} eval epochs. "
                    f"Best val_mAUC={best_val_auc:.6f} at epoch {best_epoch}."
                )
                break

    print("[train] Training complete.")
    print(f"[train] Best val_mAUC={best_val_auc:.6f} @ epoch {best_epoch}")
    print(f"[train] last checkpoint: {last_path}")
    print(f"[train] best checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
