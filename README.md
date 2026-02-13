# Hibiki Zero S2S

Real-time speech-to-speech translation deployed on [Baseten](https://baseten.co) via WebSocket.

Powered by [Kyutai's Hibiki Zero](https://github.com/kyutai-labs/hibiki-zero) — a 3B parameter model for simultaneous translation.

**Supported:** French, Spanish, Portuguese, German → English

## Architecture

```
Browser (mic) → WebSocket → Local Proxy → Baseten (GPU) → Translated audio + text
```

- **`index.html`** — Single-file frontend with Opus encoding/decoding, waveform visualization
- **`proxy.py`** — Local WebSocket proxy that injects Baseten auth header
- **`truss/`** — Baseten Truss deployment (L4 GPU, WebSocket transport)

## Quick Start

### 1. Deploy to Baseten

```bash
pip install truss
cp .env.example .env  # add your BASETEN_API_KEY
cd truss && truss push
```

### 2. Run locally

```bash
source .env
export BASETEN_API_KEY

# Start proxy (relays to Baseten with auth)
pip install websockets
python3 proxy.py &

# Serve frontend
python3 -m http.server 8080 &

open http://localhost:8080
```

### 3. Speak

Click **Start Translating**, speak in French/Spanish/Portuguese/German, hear English translation in real-time.

## WebSocket Protocol

Binary messages with 1-byte prefix:

| Prefix | Direction | Content |
|--------|-----------|---------|
| `0x00` | Server → Client | Handshake |
| `0x01` | Both | Opus audio |
| `0x02` | Server → Client | UTF-8 text |

## Config

| Setting | Value |
|---------|-------|
| GPU | Nvidia L4 (24 GB) |
| Model | `kyutai/hibiki-zero-3b-pytorch-bf16` |
| Audio | Opus @ 24kHz, 20ms frames |
| Latency | ~10-20ms per frame |

## License

Code: MIT. Model weights: [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial).
