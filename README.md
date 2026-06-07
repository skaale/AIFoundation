# AIFoundation

Hands-free voice conversation app for the ASI Foundation Church — a priest persona powered by [Supra-50M-Reasoning](https://huggingface.co/SupraLabs/Supra-50M-Reasoning), Kokoro TTS, and Whisper STT.

## Requirements

- Python 3.11+
- CUDA GPU recommended (CPU fallback supported)
- FFmpeg (for `torchaudio` / speech decoding)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser. Allow microphone access when prompted. The priest introduction plays once; listening starts after it finishes.

## CLI

```bash
python cli.py
```

## License

MIT — see [LICENSE](LICENSE).
