from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from biomass.config import ProjectConfig
from biomass.data import (
    BiomassTrainDataset,
    build_eval_transform,
    build_train_transform,
    load_train_table,
)
from biomass.metrics import weighted_r2_score
from biomass.model import BiomassModel
from biomass.utils import ensure_dir, save_json, seed_everything, setup_torch_runtime


def biomass_loss(preds: torch.Tensor, labels: torch.Tensor, beta: float) -> torch.Tensor:
    return nn.SmoothL1Loss(beta=beta)(preds, labels)


def build_optimizer(model: BiomassModel, cfg: ProjectConfig) -> optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.train["lr_backbone"]},
            {"params": head_params, "lr": cfg.train["lr_head"]},
        ],
        weight_decay=cfg.train["weight_decay"],
    )


def build_scheduler(optimizer: optim.Optimizer, total_steps: int, cfg: ProjectConfig) -> LambdaLR:
    warmup_steps = cfg.train["warmup_epochs"] * max(1, total_steps // cfg.train["epochs"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for step, (left, right, labels) in enumerate(pbar, start=1):
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            preds = model(left, right)
            loss = biomass_loss(preds, labels, cfg.train["smooth_l1_beta"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train["clip_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / step:.4f}")
    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, cfg) -> tuple[float, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    for left, right, labels in tqdm(loader, desc="Validating", leave=False):
        left = left.to(device)
        right = right.to(device)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            preds = model(left, right)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return weighted_r2_score(y_true, y_pred, cfg.target_weights)


def run_training(cfg: ProjectConfig) -> dict:
    seed_everything(cfg.seed)
    setup_torch_runtime()
    ensure_dir(cfg.path("model_dir"))
    ensure_dir(cfg.path("output_dir"))

    train_df = load_train_table(
        train_csv=cfg.path("train_csv"),
        targets=cfg.targets,
        n_folds=cfg.train["n_folds"],
        seed=cfg.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))
    fold_scores = []

    for fold in cfg.train["folds_to_train"]:
        train_data = train_df[train_df["fold"] != fold].reset_index(drop=True)
        val_data = train_df[train_df["fold"] == fold].reset_index(drop=True)
        train_dataset = BiomassTrainDataset(
            train_data,
            image_dir=cfg.path("train_image_dir"),
            targets=cfg.targets,
            transform=build_train_transform(cfg.model["img_size"]),
        )
        val_dataset = BiomassTrainDataset(
            val_data,
            image_dir=cfg.path("train_image_dir"),
            targets=cfg.targets,
            transform=build_eval_transform(cfg.model["img_size"]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train["batch_size"],
            shuffle=True,
            num_workers=cfg.train["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train["batch_size"],
            shuffle=False,
            num_workers=cfg.train["num_workers"],
            pin_memory=True,
        )
        model = BiomassModel(
            model_name=cfg.model["model_name"],
            dropout=cfg.model["dropout"],
            pretrained=cfg.model["use_pretrained"],
        ).to(device)
        optimizer = build_optimizer(model, cfg)
        total_steps = len(train_loader) * cfg.train["epochs"]
        scheduler = build_scheduler(optimizer, total_steps, cfg)
        best_r2 = -float("inf")
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(cfg.train["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg)
            val_r2, per_target = validate(model, val_loader, device, cfg)
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_epoch = epoch + 1
                stale_epochs = 0
                torch.save(model.state_dict(), cfg.path("model_dir") / f"fold{fold}_best.pth")
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.train["early_stopping_patience"]:
                    break
            print(
                f"fold={fold} epoch={epoch + 1} train_loss={train_loss:.4f} "
                f"val_r2={val_r2:.4f} per_target={np.round(per_target, 4).tolist()}"
            )
        fold_scores.append(float(best_r2))
        print(f"fold={fold} best_r2={best_r2:.4f} best_epoch={best_epoch}")
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "model_name": cfg.model["model_name"],
        "n_folds": cfg.train["n_folds"],
        "fold_scores": fold_scores,
        "mean_cv": float(np.mean(fold_scores)),
        "std_cv": float(np.std(fold_scores)),
    }
    save_json(summary, cfg.path("output_dir") / "training_summary.json")
    return summary
