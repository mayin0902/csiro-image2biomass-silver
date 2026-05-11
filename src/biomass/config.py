from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def train(self) -> dict[str, Any]:
        return self.raw["train"]

    @property
    def inference(self) -> dict[str, Any]:
        return self.raw["inference"]

    @property
    def metric(self) -> dict[str, Any]:
        return self.raw["metric"]

    @property
    def targets(self) -> list[str]:
        return list(self.model["targets"])

    @property
    def target_weights(self) -> list[float]:
        weight_map = self.metric["weighted_r2_weights"]
        return [float(weight_map[target]) for target in self.targets]

    def path(self, key: str) -> Path:
        return Path(self.paths[key])


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(raw=raw)
