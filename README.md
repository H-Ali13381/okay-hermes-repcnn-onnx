---
license: apache-2.0
library_name: onnx
pipeline_tag: audio-classification
tags:
- wake-word
- keyword-spotting
- onnx
- audio
- edge-ai
- repcnn
- wavlm
- okay-hermes
---

# Okay Hermes Wake-Word Model

[![Hugging Face downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FNeopabo%2Fokay-hermes-repcnn-onnx&query=%24.downloads&label=HF%20downloads&color=blue)](https://huggingface.co/Neopabo/okay-hermes-repcnn-onnx)

Compact ONNX wake-word model for detecting **“Okay Hermes”** in short local audio windows.

This is the current WavLM-teacher → merged RepCNN-student export. It is intended for local assistants, desktop agents, and voice-enabled tools that need a lightweight wake trigger before starting a larger speech or reasoning pipeline.

## Links

- Hugging Face: [https://huggingface.co/Neopabo/okay-hermes-repcnn-onnx](https://huggingface.co/Neopabo/okay-hermes-repcnn-onnx)
- GitHub: [https://github.com/H-Ali13381/okay-hermes-repcnn-onnx](https://github.com/H-Ali13381/okay-hermes-repcnn-onnx)

## Repository contents

- `wakeword.onnx` — deployable merged RepCNN ONNX model
- `wakeword.json` — export metadata and recommended threshold
- `config.json` — small Hub config for loaders and download statistics
- `README.md` — model card and usage notes
- `LICENSE` — Apache-2.0 license

## What this model does

- Takes a 3-second mono audio window.
- Returns a direct probability that the wake phrase **“Okay Hermes”** is present.
- Runs locally with ONNX Runtime.
- Does not transcribe speech.
- Does not identify speakers.
- Does not send audio anywhere by itself.

## Input

- **Name:** `waveform`
- **Shape:** `(batch, time)`; use exactly 48,000 samples per row for a 3-second window at 16 kHz
- **Type:** `float32`
- **Audio:** mono PCM, 16 kHz, 3 seconds

Audio should be resampled to 16 kHz mono and padded or cropped to exactly 48,000 samples per window.

## Output

- **Name:** `score`
- **Shape:** `(batch,)`
- **Type:** `float32`
- **Meaning:** wake-word probability for each input window

The output is already a probability. Do **not** apply another sigmoid unless you have changed the exported graph.

Recommended starting threshold:

```text
0.6973556280136108
```

Basic trigger rule:

```text
trigger = score >= 0.6973556280136108
```

For always-on use, apply smoothing or debouncing across overlapping audio windows. A practical starting point is 3-second windows every 0.5 seconds with 2 consecutive windows above threshold.

## Python example

```python
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

repo_id = "Neopabo/okay-hermes-repcnn-onnx"

# Fetch config first so normal Hub usage touches a counted metadata file.
config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

model_path = hf_hub_download(repo_id=repo_id, filename=config["onnx_file"])

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

# Replace this with real mono 16 kHz audio, exactly 3 seconds long.
waveform = np.zeros((48000,), dtype=np.float32)

score = session.run(
    [config["output_name"]],
    {config["input_name"]: waveform[None, :]},
)[0][0]

trigger = float(score) >= float(config["recommended_threshold"])
print(float(score), trigger)
```

## Technical summary

- Wake phrase: **Okay Hermes**
- Backend: `wavlm-repcnn`
- Inference model: merged / reparameterized RepCNN
- Teacher model: `microsoft/wavlm-base`
- Audio frontend: included in the ONNX graph; callers provide waveform audio directly
- Output: direct probability named `score`
- ONNX input shape: dynamic time axis named `time`; use 48,000 samples for the documented 3-second window
- Recommended threshold: `0.6973556280136108`
- Export metadata EER: `0.0`
- ONNX size: `224807` bytes
- ONNX SHA256: `f022856f17916f5c7b2d8041f44308f889f292d92be12b71e1fe3ee49bd0a0fc`

Training used hard-negative and partial-negative examples to reduce accidental activations. Field performance still depends on microphone quality, room acoustics, accents, background noise, and trigger/debounce logic.

## Checksum

```text
wakeword.onnx SHA256: f022856f17916f5c7b2d8041f44308f889f292d92be12b71e1fe3ee49bd0a0fc
```

## Limitations

- Wake-word detection only; not general speech recognition.
- Not speaker verification or identity recognition.
- False accepts and false rejects are possible.
- Thresholds may need adjustment for each deployment environment.
- No training audio or source datasets are included in this repository.

## Privacy

The model runs locally and only processes audio supplied by the surrounding application. Privacy depends on how that application captures, buffers, logs, stores, or transmits audio.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
