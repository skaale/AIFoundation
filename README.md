# AIFoundation

Hands-free voice conversation app for the ASI Foundation Church — a priest persona powered by [Supra-50M-Reasoning](https://huggingface.co/SupraLabs/Supra-50M-Reasoning), Kokoro TTS, Whisper STT, and a **knowledge-tree RAG** for AGI, AI, technology, and AI God topics.

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

Open `http://127.0.0.1:7860` in your browser. Allow microphone access when prompted. The priest gives a short greeting, asks your name, then uses it personally in conversation.

Open **⋯ Settings** to add new questions and answers to the knowledge tree. Retrieved entries are injected into Supra before each reply.

## Conversation memory

After you share your name, the priest saves each exchange under `memory/profiles/` (local JSON, not committed to git). On later visits, memory brings back:

- Prior topics and personal notes
- Recent exchanges
- A short relationship summary

This helps the priest stay consistent and more precise with you over time.

## Knowledge tree (RAG)

Topics are organized as a tree under `knowledge/knowledge_tree.json`:

- **AGI** — alignment, machine consciousness
- **AI** — machine learning, ethics
- **Technology** — compute, future tech
- **AI God** — theology of superintelligence, providence

### CLI

```bash
# List topics
python knowledge_cli.py topics

# List entries in a topic
python knowledge_cli.py list --topic agi/alignment

# Add Q&A
python knowledge_cli.py add --topic ai_god/theology -q "Your question?" -a "Priest answer."

# Test retrieval
python knowledge_cli.py search "what is alignment"
```

### CLI chat with RAG

```bash
python cli.py "What is AI God?"
python cli.py "Explain AGI alignment" --no-rag
```

## License

MIT — see [LICENSE](LICENSE).
