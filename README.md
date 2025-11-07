# A Low-latency Voice Bot built with Modal and Pipecat 

A real-time conversational AI bot powered by [Pipecat](https://github.com/pipecat-ai/pipecat) and deployed on [Modal](https://modal.com). This project features an interactive RAG (Retrieval-Augmented Generation) system with real-time speech-to-speech interaction.

[Blog post](https://modal.com/blog/low-latency-voice-bot)

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:modal-projects/open-source-av-ragbot.git
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

Go to [modal.com](modal.com) and make an account if you don't have one.

```bash
# Authenticate your Modal installation
modal setup
```

### 4. Set Up Client

## Install dependencies

```bash
cd client
npm i
```

## Build

```bash
npm run build

# return to root dir
cd ..
```

## Deployment

### Deploy All Services

The project consists of multiple Modal services that need to be deployed:

```bash
# From the root dir of the project

# Deploy an LLM Service

# etiher VLLM inference server for optimized TTFT
modal deploy -m server.llm.vllm_server

# or use SGLang server for 
# aster cold starts with GPU snapshots
modal deploy -m server.llm.sglang_server

# Deploy Parakeet STT service
modal deploy -m server.stt.parakeet_stt

# Deploy Kokoro TTS service
modal deploy -m server.tts.kokoro_tts

# Deploy main bot application with frontend
modal deploy -m app
```

### Warmup Snapshots
We can speed up the cold start time of our bot (this is more important) and our Parakeet and LLM service (if using SGLang) using snapshots. However this leads to extra start up time for the first few containers when the apps are (re-)deployed. To warmup snapshots, you can run these files as Python scripts.

```bash
python -m server.stt.parakeet_stt

python -m server.llm.sglang_server

python -m app
```

