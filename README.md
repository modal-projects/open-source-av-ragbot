# A Low-latency Voice Bot built with Modal and Pipecat 

A real-time conversational AI bot powered by [Pipecat](https://github.com/pipecat-ai/pipecat) and deployed on [Modal](https://modal.com). This project features an interactive RAG (Retrieval-Augmented Generation) system with real-time speech-to-speech interaction.

[Blog post](https://modal.com/blog/low-latency-voice-bot)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd open-source-av-ragbot
```

### 2. Set Up Python Environment

This project uses [uv](https://github.com/astral-sh/uv) for Python package management:

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 3. Configure Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup
```

### 4. Set Up Client

## Point the client at your deploy API URL

```bash
mv env.example .env

# Set VITE_API_URL to your Modal server URL deployment
# e.g. https://{YOUR_MODAL_ORG}--moe-and-dal-ragbot-bot-server.modal.run
```

## Install dependencies

```bash
npm i
```

## Build or Run 

```bash
npm run build

# or

npm run dev
```

## Deployment

### Deploy All Services

The project consists of multiple Modal services that need to be deployed:

```bash
# Deploy VLLM inference server
modal deploy -m server.llm.vllm_server

# Deploy Parakeet STT service
modal deploy -m server.stt.parakeet_stt

# Deploy Kokoro TTS service
modal deploy -m server.tts.kokoro_tts


# Deploy main bot application with frontend
modal deploy -m app
```

### Warmup Snapshots
We can speed up the cold start time of our bot (this is more important) and our Parakeet service using snapshots. However this leads to extra start up time for the first few containers when the apps are (re-)deployed. To warmup snapshots, you can run these files as Python scripts.

```bash
python -m server.stt.parakeet_stt

python -m app
```

