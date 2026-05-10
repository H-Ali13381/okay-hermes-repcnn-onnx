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
---

# Okay Hermes Wake-Word Model

A compact ONNX wake-word model for detecting the phrase **“Okay Hermes”** in short audio windows.

This release is intended for local assistants, desktop agents, and voice-enabled applications that need a lightweight wake trigger before starting a larger speech or reasoning pipeline.

## Links

- Hugging Face: [https://huggingface.co/Neopabo/okay-hermes-repcnn-onnx](https://huggingface.co/Neopabo/okay-hermes-repcnn-onnx)
- GitHub: [https://github.com/H-Ali13381/okay-hermes-repcnn-onnx](https://github.com/H-Ali13381/okay-hermes-repcnn-onnx)

## What this model does

- Takes a 3-second mono audio window.
- Returns a probability that the wake phrase **“Okay Hermes”** is present.
- Runs locally with ONNX Runtime.
- Does not transcribe speech.
- Does not identify speakers.
- Does not send audio anywhere by itself.

## Repository contents

- `retrained_20260510_165910_folded.onnx` — deployable ONNX model
- `README.md` — model card and usage notes
- `LICENSE` — Apache-2.0 license

## Input

- **Name:** `waveform`
- **Shape:** `(batch, 48000)`
- **Type:** `float32`
- **Audio:** mono PCM, 16 kHz, 3 seconds

Audio should be resampled to 16 kHz mono and padded or cropped to exactly 48,000 samples per window.

## Output

- **Name:** `probabilities`
- **Shape:** `(batch,)`
- **Type:** `float32`
- **Meaning:** wake-word probability for each input window

Recommended starting threshold:

```text
0.4112943708896637
```

Basic trigger rule:

```text
trigger = probability >= 0.4112943708896637
```

For production use, apply smoothing or debouncing across overlapping audio windows instead of triggering from a single score.

## Python example

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "retrained_20260510_165910_folded.onnx",
    providers=["CPUExecutionProvider"],
)

# Replace this with real mono 16 kHz audio, exactly 3 seconds long.
waveform = np.zeros((48000,), dtype=np.float32)

probability = session.run(
    ["probabilities"],
    {"waveform": waveform[None, :]},
)[0][0]

trigger = float(probability) >= 0.4112943708896637
print(float(probability), trigger)
```

## Technical summary

This artifact demonstrates a practical research-to-deployment workflow for wake-word detection:

- A larger WavLM-based teacher model was used to train a compact RepCNN student.
- The final RepCNN was folded into a simpler inference graph for deployment.
- Audio preprocessing is included in the ONNX graph, so applications can provide waveform audio directly.
- Training included hard-negative and partial-negative examples to reduce accidental activations.
- The exported ONNX model was validated with ONNX Runtime.

The result is a small local inference artifact suitable for desktop and edge-style assistant workflows.

## Evaluation snapshot

- ONNX size: about 359 KB
- Operating threshold: `0.4112943708896637`
- Target false-accept rate: `0.001`
- False-reject rate at target FAR: about `1.08%`
- Partial false-accept rate: about `0.56%`

These numbers are a starting point, not a universal guarantee. Real-world behavior depends on microphone quality, background noise, accents, room acoustics, and trigger logic.

## Checksum

```text
ONNX SHA256: e705d3af445ab38666b06a1f475339ca47b3f8645e6d53d056a11db0a7a9fb19
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
