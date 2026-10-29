# Transformer foundations (teaching code)

These small components show the ideas behind Transformers. They are deliberately separate from the production path, which will use pretrained models.

## What is a token?
A token is one item in a sequence. For text it can be a character or word piece; for images it can be a small patch; for audio it can be a time-frequency patch; for video it can be a space-time tubelet.

```text
text:   "cat"       -> [c, a, t] -> token IDs -> vectors
image:  [pixels]    -> [patches] -> projected vectors + CLS
sound:  [waveform]  -> [log-mel spectrogram] -> patches -> vectors
video:  [frames]   -> [space-time tubelets] -> vectors
```

## Attention in plain language
For each token, attention asks: which other tokens are useful for understanding this one? It creates query (Q), key (K), and value (V) vectors. Similarity between Q and K gives weights; those weights mix V vectors.

```text
Q x K^T / sqrt(head_size) -> mask -> softmax -> weights x V -> output
```

A padding mask hides unused tokens. A causal mask stops a token from looking at future tokens, useful for next-token generation. Multi-head attention repeats this process in several learned subspaces.

## Transformer encoder block

```text
input -> LayerNorm -> Multi-head attention -> add residual
      -> LayerNorm -> GELU feed-forward -> add residual -> output
```

LayerNorm stabilizes activations. GELU is a smooth activation function. Residual connections preserve the input signal and help gradients flow. Positional encoding adds information about sequence order; without it, self-attention alone is permutation-equivariant.

## Run the checks

```bash
pytest -q
python -m foundations.train_smoke --steps 10
```

The smoke-training script uses random synthetic data only. It proves that the tiny models run and gradients update; it is **not** a dataset training result and must not be reported as accuracy. Training on public CIFAR-10 / ESC-50 subsets and evaluating them will be a separate measured experiment.

## What is built here vs pretrained models at scale?

| This teaching code | Pretrained model in the production path |
|---|---|
| Small attention and encoder blocks | Millions/billions of parameters trained on large corpora |
| Character IDs, image patches, log-mel audio patches, video tubelets | Model-specific tokenizers and preprocessing |
| Tiny randomly initialized classifier | Learned visual/audio/speech representations |
| CPU-sized examples to inspect shapes | Zero-shot predictions or embeddings, evaluated on task data |
