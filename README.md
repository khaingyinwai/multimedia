# Multimodal Recognition Prototype — Checkpoint 1

This checkpoint creates the specification, demonstration labels, configuration, and config validator. It does **not** yet run pretrained models or make real predictions.

## Use from a Mac with GitHub Codespaces

1. Download and extract the ZIP.
2. In GitHub, create a repository, then upload the extracted files/folders.
3. Open **Code → Codespaces → Create codespace**.
4. In the Codespaces terminal run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m core.config
pytest -q
```

Expected config output starts with `CONFIG OK`. Tests should report `3 passed`.

## Run locally on a Mac instead

Install Python 3.11 or 3.12, extract the project, open Terminal in the project folder, and run the same commands above. No GPU is needed for this checkpoint.

## Model choices verified against Hugging Face model pages

- Image: `openai/clip-vit-large-patch14` (large download; use `openai/clip-vit-base-patch32` if CPU/RAM is limited).
- General audio: `laion/larger_clap_general`.
- Speech: `openai/whisper-tiny`.

No model weights are downloaded in Checkpoint 1.

## Files
- `SPEC.md`: scope, assumptions, task, failure modes, targets and privacy notes.
- `config.yaml`: tunable values and model IDs.
- `data/labels.json`: demo label catalog with `unknown` and `uncertain`.
- `core/config.py`: config validation CLI.
- `tests/test_config.py`: offline tests.

## Limits
The taxonomy is generic until the actual use case is described. Thresholds and success criteria are targets, not measured accuracy. Do not use demo labels for high-stakes decisions.

## Checkpoint 2: Transformer foundations

The `foundations/` folder contains CPU-sized teaching implementations of scaled dot-product attention, multi-head attention, padding/causal masks, LayerNorm, GELU feed-forward layers, residual connections, positional encoding, character tokenization, image patches, log-mel audio patches, video tubelets, and tiny ViT/audio classifier models.

After installing the updated pinned requirements, run:

```bash
python -m core.config
python -m pytest -q
python -m foundations.train_smoke --steps 10
```

The smoke-training command uses synthetic random data only. It is not evidence of model accuracy. The pretrained production path is not wired up yet.
