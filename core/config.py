"""Configuration loading and validation.

Run: python -m core.config
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class ModelConfig(BaseModel):
    provider_order: list[str] = Field(default_factory=list)
    model_id: str | None = None
    alternative_model_id: str | None = None
    enabled: bool = True


class AppConfig(BaseModel):
    name: str
    environment: str
    mock_mode: bool
    max_upload_mb: int = Field(gt=0, le=500)
    cache_dir: str
    data_dir: str
    log_level: str


class FusionConfig(BaseModel):
    min_confidence: float = Field(ge=0, le=1)
    uncertain_below: float = Field(ge=0, le=1)
    top_k: int = Field(ge=1, le=20)
    weights: dict[str, float]
    agreement_bonus: dict[str, float]
    reject_label_id: str
    uncertain_label_id: str

    @field_validator("weights")
    @classmethod
    def weights_must_be_nonnegative(cls, value):
        if any(weight < 0 for weight in value.values()):
            raise ValueError("fusion weights must be non-negative")
        return value


class RootConfig(BaseModel):
    app: AppConfig
    models: dict[str, Any]
    fusion: FusionConfig
    rate_limits: dict[str, Any]
    generators: dict[str, Any]
    evaluation: dict[str, Any]


def load_config(path: str | Path = "config.yaml") -> RootConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    config = RootConfig.model_validate(raw)
    if config.fusion.uncertain_below < config.fusion.min_confidence:
        raise ValueError("fusion.uncertain_below must be >= fusion.min_confidence")
    for modality in ("image", "sound", "speech", "video", "text"):
        if modality not in config.models:
            raise ValueError(f"models.{modality} is required")
    return config


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.yaml")
    try:
        config = load_config(path)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"CONFIG INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"CONFIG OK: {config.app.name}")
    print(f"Mock mode: {config.app.mock_mode}")
    print(f"Modalities configured: {', '.join(config.models.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
