import json
from pathlib import Path
import pytest
from core.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    config = load_config(ROOT / "config.yaml")
    assert config.app.mock_mode is True
    assert {"image", "sound", "speech", "video", "text"} <= set(config.models)
    assert config.fusion.top_k == 3


def test_label_catalog_has_unknown_and_uncertain():
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    ids = {item["id"] for item in labels["labels"]}
    assert "unknown" in ids
    assert "uncertain" in ids
    assert len(ids) == len(labels["labels"])


def test_invalid_config_rejected(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("app: []")
    with pytest.raises(Exception):
        load_config(bad)
