from __future__ import annotations

import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset


def load_train_table(train_csv: str | Path, targets: list[str], n_folds: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df_wide = (
        df.pivot_table(
            index=["image_id", "image_path"],
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )
    for col in targets:
        if col not in df_wide.columns:
            df_wide[col] = 0.0
    df_wide["total_bin"] = pd.qcut(df_wide["Dry_Total_g"], q=5, labels=False, duplicates="drop")
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df_wide["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        splitter.split(df_wide, df_wide["total_bin"], groups=df_wide["image_id"])
    ):
        df_wide.loc[val_idx, "fold"] = fold
    return df_wide


def load_test_table(test_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    return df.drop_duplicates("image_id")[["image_id", "image_path"]].reset_index(drop=True)


def build_train_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def build_eval_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class BiomassTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str | Path, targets: list[str], transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_dir / os.path.basename(row["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        mid = w // 2
        left = image[:, :mid]
        right = image[:, mid:]
        left = self.transform(image=left)["image"]
        right = self.transform(image=right)["image"]
        labels = torch.tensor(row[self.targets].to_numpy(dtype=np.float32))
        return left, right, labels


class BiomassTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str | Path, img_size: int):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        left = image.crop((0, 0, h, h)).resize((self.img_size, self.img_size))
        right = image.crop((w - h, 0, w, h)).resize((self.img_size, self.img_size))
        left_arr = (np.asarray(left).astype(np.float32) / 255.0 - self.mean) / self.std
        right_arr = (np.asarray(right).astype(np.float32) / 255.0 - self.mean) / self.std
        left_tensor = torch.from_numpy(left_arr.transpose(2, 0, 1)).float()
        right_tensor = torch.from_numpy(right_arr.transpose(2, 0, 1)).float()
        return left_tensor, right_tensor, row.to_dict()


def collate_test(batch):
    lefts = torch.stack([item[0] for item in batch])
    rights = torch.stack([item[1] for item in batch])
    infos = [item[2] for item in batch]
    return lefts, rights, infos
