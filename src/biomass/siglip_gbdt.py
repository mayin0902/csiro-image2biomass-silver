from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


def split_image(image: np.ndarray, patch_size: int = 520, overlap: int = 16) -> list[np.ndarray]:
    h, w, _ = image.shape
    stride = patch_size - overlap
    patches = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size)
            x1 = max(0, x2 - patch_size)
            patches.append(image[y1:y2, x1:x2, :])
    return patches


def compute_siglip_embeddings(model_path: str, df: pd.DataFrame, image_column: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SigLIP Embeddings", leave=False):
        try:
            image = cv2.imread(str(row[image_column]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            patches = [Image.fromarray(p) for p in split_image(image)]
            inputs = processor(images=patches, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            embeddings.append(features.mean(dim=0).cpu().numpy())
        except Exception:
            embeddings.append(np.zeros(1152, dtype=np.float32))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(embeddings)


def generate_semantic_features(embeddings: np.ndarray, model_path: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    concepts = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"],
    }
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concepts.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            concept_vectors[name] = text_features.mean(dim=0, keepdim=True)
    image_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    image_tensor = image_tensor / image_tensor.norm(p=2, dim=-1, keepdim=True)
    scores = {}
    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(image_tensor, vec.T).cpu().numpy().flatten()
    scores_df = pd.DataFrame(scores)
    scores_df["ratio_greenness"] = scores_df["green"] / (scores_df["green"] + scores_df["dead"] + 1e-6)
    scores_df["ratio_clover"] = scores_df["clover"] / (scores_df["clover"] + scores_df["grass"] + 1e-6)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores_df.values


class SupervisedEmbeddingEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.80, random_state=42)
        self.pls = PLSRegression(n_components=8, scale=False)
        self.gmm = GaussianMixture(n_components=6, covariance_type="diag", random_state=42)
        self.pls_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SupervisedEmbeddingEngine":
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        if y is not None:
            self.pls.fit(X_scaled, y)
            self.pls_fitted = True
        return self

    def transform(self, X: np.ndarray, X_semantic: np.ndarray | None = None) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        features = [self.pca.transform(X_scaled)]
        if self.pls_fitted:
            features.append(self.pls.transform(X_scaled))
        features.append(self.gmm.predict_proba(X_scaled))
        if X_semantic is not None:
            sem = (X_semantic - X_semantic.mean(axis=0)) / (X_semantic.std(axis=0) + 1e-6)
            features.append(sem)
        return np.hstack(features)


def train_gbdt_cv(train_df, test_df, target_names, target_max, emb_cols, sem_train, sem_test):
    model_specs = [
        (HistGradientBoostingRegressor, {"max_depth": 6, "learning_rate": 0.05, "max_iter": 400}),
        (GradientBoostingRegressor, {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3}),
        (LGBMRegressor, {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "verbose": -1}),
        (
            CatBoostRegressor,
            {"iterations": 400, "learning_rate": 0.03, "depth": 6, "verbose": False},
        ),
    ]
    target_max_arr = np.array([target_max[t] for t in target_names], dtype=np.float32)
    y_pred_test = np.zeros((len(test_df), len(target_names)), dtype=np.float32)
    X_train = train_df[emb_cols].values.astype(np.float32)
    X_test = test_df[emb_cols].values.astype(np.float32)
    y_train = train_df[target_names].values.astype(np.float32)
    n_splits = int(train_df["fold"].nunique())

    for fold in range(n_splits):
        train_mask = train_df["fold"] != fold
        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask] / target_max_arr
        sem_tr = sem_train[train_mask]
        engine = SupervisedEmbeddingEngine().fit(X_tr, y=y_tr)
        x_tr = engine.transform(X_tr, X_semantic=sem_tr)
        x_te = engine.transform(X_test, X_semantic=sem_test)
        fold_pred = np.zeros((len(test_df), len(target_names)), dtype=np.float32)
        model_count = np.zeros(len(target_names), dtype=np.float32)
        for model_cls, params in model_specs:
            for idx, target in enumerate(target_names):
                if target == "Dry_Clover_g":
                    continue
                model = model_cls(**params)
                model.fit(x_tr, y_tr[:, idx])
                fold_pred[:, idx] += model.predict(x_te) * target_max_arr[idx]
                model_count[idx] += 1
        for idx in range(len(target_names)):
            if model_count[idx] > 0:
                fold_pred[:, idx] /= model_count[idx]
        y_pred_test += fold_pred
    return y_pred_test / n_splits


def normalize_split_paths(df: pd.DataFrame, data_root: str, image_column: str) -> pd.DataFrame:
    data_root = str(data_root)
    fixed = df.copy()
    fixed[image_column] = fixed[image_column].apply(
        lambda p: p if str(p).startswith("/") else os.path.join(data_root, "train", os.path.basename(str(p)))
    )
    return fixed
