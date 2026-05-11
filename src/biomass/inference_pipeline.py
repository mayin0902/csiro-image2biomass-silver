from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from biomass.config import ProjectConfig
from biomass.data import BiomassTestDataset, collate_test, load_test_table
from biomass.model import BiomassModel
from biomass.siglip_gbdt import (
    compute_siglip_embeddings,
    generate_semantic_features,
    normalize_split_paths,
    train_gbdt_cv,
)
from biomass.utils import ensure_dir, seed_everything, setup_torch_runtime


@torch.no_grad()
def predict_dino(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for left, right, _ in tqdm(loader, desc="DINO Inference", leave=False):
        left = left.to(device)
        right = right.to(device)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(left, right)
        preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def apply_dino_postprocess(dino_df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    infer_cfg = cfg.inference
    dino_df = dino_df.copy()
    dino_df["Dry_Clover_g"] *= infer_cfg["clover_scale"]
    high_mask = dino_df["Dry_Dead_g"] > infer_cfg["dead_high_threshold"]
    low_mask = dino_df["Dry_Dead_g"] < infer_cfg["dead_low_threshold"]
    dino_df.loc[high_mask, "Dry_Dead_g"] *= infer_cfg["dead_high_scale"]
    dino_df.loc[low_mask, "Dry_Dead_g"] *= infer_cfg["dead_low_scale"]
    dino_df["GDM_g"] = dino_df["Dry_Green_g"] + dino_df["Dry_Clover_g"]
    dino_df["Dry_Total_g"] = dino_df["GDM_g"] + dino_df["Dry_Dead_g"]
    for target in cfg.targets:
        dino_df[target] = dino_df[target].clip(lower=0.0)
    return dino_df


def run_inference(cfg: ProjectConfig) -> pd.DataFrame:
    seed_everything(cfg.seed)
    setup_torch_runtime()
    ensure_dir(cfg.path("output_dir"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_wide = load_test_table(cfg.path("test_csv"))
    test_loader = DataLoader(
        BiomassTestDataset(test_wide, image_root=cfg.path("data_root"), img_size=cfg.model["img_size"]),
        batch_size=cfg.inference["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_test,
    )

    all_fold_preds = []
    for fold in range(cfg.inference["n_folds"]):
        model_path = cfg.path("model_dir") / f"fold{fold}_best.pth"
        if not model_path.exists():
            continue
        model = BiomassModel(
            model_name=cfg.model["model_name"],
            dropout=cfg.model["dropout"],
            pretrained=False,
        ).to(device)
        state_dict = torch.load(model_path, map_location=device)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        all_fold_preds.append(predict_dino(model, test_loader, device))
        del model, state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not all_fold_preds:
        raise FileNotFoundError("No DINO fold checkpoints were found.")

    dino_preds = np.mean(all_fold_preds, axis=0)
    dino_df = test_wide.copy()
    for idx, target in enumerate(["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]):
        dino_df[target] = dino_preds[:, idx]
    dino_df = apply_dino_postprocess(dino_df, cfg)

    split_df = pd.read_csv(cfg.path("split_csv"))
    split_df = split_df[[c for c in split_df.columns if not c.startswith("emb")]]
    split_df = normalize_split_paths(split_df, data_root=str(cfg.path("data_root")), image_column="image_path")
    test_siglip = test_wide.copy()
    test_siglip["image_path"] = test_siglip["image_path"].apply(lambda p: str(Path(cfg.path("data_root")) / p))

    train_embeddings = compute_siglip_embeddings(cfg.paths["siglip_model_path"], split_df, "image_path")
    test_embeddings = compute_siglip_embeddings(cfg.paths["siglip_model_path"], test_siglip, "image_path")
    train_semantic = generate_semantic_features(train_embeddings, cfg.paths["siglip_model_path"])
    test_semantic = generate_semantic_features(test_embeddings, cfg.paths["siglip_model_path"])

    emb_cols = [f"emb_{i}" for i in range(train_embeddings.shape[1])]
    split_df = split_df.copy()
    test_siglip = test_siglip.copy()
    split_df[emb_cols] = train_embeddings
    test_siglip[emb_cols] = test_embeddings

    siglip_targets = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
    siglip_preds = train_gbdt_cv(
        train_df=split_df,
        test_df=test_siglip,
        target_names=siglip_targets,
        target_max=cfg.model["target_max"],
        emb_cols=emb_cols,
        sem_train=train_semantic,
        sem_test=test_semantic,
    )
    siglip_df = pd.DataFrame(siglip_preds, columns=siglip_targets)
    final_df = pd.DataFrame({"image_id": test_wide["image_id"], "image_path": test_wide["image_path"]})
    final_df["Dry_Green_g"] = (
        cfg.inference["dino_weight"] * dino_df["Dry_Green_g"].to_numpy()
        + cfg.inference["siglip_weight"] * siglip_df["Dry_Green_g"].to_numpy()
    )
    final_df["Dry_Dead_g"] = (
        cfg.inference["dino_weight"] * dino_df["Dry_Dead_g"].to_numpy()
        + cfg.inference["siglip_weight"] * siglip_df["Dry_Dead_g"].to_numpy()
    )
    final_df["Dry_Clover_g"] = dino_df["Dry_Clover_g"].to_numpy()
    final_df["GDM_g"] = final_df["Dry_Green_g"] + final_df["Dry_Clover_g"]
    final_df["Dry_Total_g"] = final_df["GDM_g"] + final_df["Dry_Dead_g"]
    for target in cfg.targets:
        final_df[target] = final_df[target].clip(lower=0.0)
    return final_df
