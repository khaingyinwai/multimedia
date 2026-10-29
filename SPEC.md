# SPEC — Generic Multimodal Recognition Prototype

## Status
Checkpoint 1: specification, demonstration label catalog, configuration, and config validation.
This is not yet a validated domain-specific classifier.

## Assumptions
- The final recognition domain has not yet been supplied, so this checkpoint uses generic demonstration labels.
- Inputs may contain any subset of image, sound/audio, speech, video, and text.
- The default task is open-set recognition: select a known label only when evidence is strong enough; otherwise return `UNKNOWN` or `UNCERTAIN`.
- Video is analyzed by sampling frames and separately extracting its audio track.
- CPU-first development is preferred. Large models may be slow and may exceed the memory budget of some Codespaces.

## Modalities and contribution
- Image: visual content from one still image; CLIP compares image and text embeddings.
- Sound: non-speech acoustic events, music, and general sound; CLAP compares audio and text embeddings.
- Speech: words spoken in an audio/video track; Whisper transcribes speech, then text logic may use the transcript.
- Video: temporal context from multiple sampled frames plus its audio track. A sampled-frame prototype can miss brief events.
- Text: direct user text or a speech transcript; useful as evidence, but does not independently validate visual/audio claims.

## Task type
Recognition/classification with open-set rejection. Results are combined with deterministic rules. The LLM, if added later, may explain structured evidence but must never make the final decision.

## Demonstration labels
The included labels are intentionally generic and are only for plumbing/demo tests:
- animal
- person
- vehicle
- indoor_scene
- outdoor_scene
- document_or_screen
- music
- speech
- alarm_or_siren
- machinery
- unknown
- uncertain

Do not use this taxonomy for medical, safety-critical, financial, identity, or official decisions. Replace it after the user describes the actual use case.

## Model choices
- Image: `openai/clip-vit-large-patch14` is the requested default and is a real Hugging Face model ID. It is large (model files shown as multi-GB); CPU use may be slow. Alternative: `openai/clip-vit-base-patch32`, smaller and usually more practical for CPU, at a likely quality trade-off.
- Sound: `laion/larger_clap_general` is a real Hugging Face model ID for zero-shot audio/text similarity. Alternative: `laion/clap-htsat-fused`, another CLAP checkpoint.
- Speech: `openai/whisper-tiny` is a real multilingual speech-recognition model. Alternative: `openai/whisper-base`, potentially better recognition at higher compute cost.
- Video: sample frames and run the image branch, then extract audio and run sound/speech branches. Alternative: a dedicated video-text model such as X-CLIP, with greater model/runtime complexity.
- Text explanation: template explanation first. Optional LLM provider can be added later; it must receive only structured agent results.

## Success criteria (targets, not measured results)
The following are engineering acceptance targets for a future domain-specific golden set, not claimed results:
- Unit tests for config and deterministic fusion: 100% pass.
- API mock-mode smoke tests: 100% pass.
- Per-modality accuracy: target >= 0.80 on a domain-approved held-out set; must report actual score and sample count.
- Unknown rejection: target >= 0.80 recall on a separate unknown set; must report actual score and sample count.
- Deliberately mismatched modality pairs: target returns `CONFLICT` >= 0.90 of tested cases.
- p95 latency: record actual latency for each modality and environment; establish target after measuring Codespaces hardware.
- Zero private or confidential files in public test fixtures.

## Failure modes
- Low resolution, blur, darkness, noise, silence, clipping, compression, or unsupported file types.
- A modality is missing, corrupt, or times out.
- Modalities disagree or refer to different subjects.
- The input is outside the label catalog.
- Zero-shot similarities are weak or misleading; raw similarity is not a calibrated probability.
- Speech recognition may mishear accents, names, technical terms, or Myanmar-language speech.
- Video frame sampling may miss short actions and order-dependent events.
- A model download fails, a provider rate-limits requests, or CPU memory/time is insufficient.

## Decision policy
- `MATCH`: usable modalities agree on the same label.
- `PARTIAL`: some evidence overlaps in top-k or label family, but not enough for a full match.
- `CONFLICT`: usable modalities disagree without meaningful overlap.
- `SINGLE_MODALITY`: only one modality supplies usable evidence.
- `UNCERTAIN`: no label clears the configured confidence threshold or evidence quality is insufficient.
- `UNKNOWN`: evidence appears out of the known label set; this is a label/outcome, not a guarantee of perfect open-set detection.

Fusion must be a pure deterministic function, tested separately from model inference. Similarity scores must not be presented as calibrated probabilities unless calibration is measured.

## Privacy and safety
Do not upload confidential organizational data to GitHub, a public repository, or an unapproved cloud Codespace. Use public/test files until data handling, retention, access control, and model licenses have been reviewed.
