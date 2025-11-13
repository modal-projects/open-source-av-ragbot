# 🎤 Rafiki - Swahili Voice AI Bot

A low-latency, conversational AI assistant that speaks Swahili fluently. Built with custom fine-tuned models and deployed on Modal for ~1 second voice-to-voice latency.

![Architecture](https://img.shields.io/badge/Architecture-Modal%20%2B%20Pipecat-blue)
![Language](https://img.shields.io/badge/Language-Swahili-green)
![Latency](https://img.shields.io/badge/Latency-~1%20second-brightgreen)

## 🌟 Features

- **Natural Swahili Conversations**: Powered by Aya-101, a 13B parameter multilingual model with excellent Swahili support
- **Custom Swahili TTS**: Uses `Nadhari/swa-csm-1b`, a fine-tuned Swahili voice model with multiple speaker options
- **Fast Speech Recognition**: Meta's Omnilingual ASR (CTC-1B) provides 16-96x faster-than-real-time transcription
- **Low Latency**: Achieves ~1 second voice-to-voice latency using Modal Tunnels and optimized inference
- **Modular Architecture**: Easy to swap models and customize behavior
- **WebRTC Transport**: Peer-to-peer audio streaming for minimal network overhead

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT (React)                          │
│              SmallWebRTC Transport (P2P)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    /offer endpoint
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   MODAL APP                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Frontend (React SPA)                                 │   │
│  │ - Speaker ID selection                               │   │
│  │ - Real-time transcription display                    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Bot Pipeline (Pipecat)                               │   │
│  │ - WebRTC ↔ VAD ↔ STT ↔ LLM ↔ TTS ↔ WebRTC         │   │
│  └──────────────────────────────────────────────────────┘   │
└───────┬──────────────────┬─────────────────┬────────────────┘
        │                  │                 │
   Modal Tunnels      Modal Tunnels     Modal Tunnels
        │                  │                 │
┌───────▼──────────┐ ┌─────▼──────────┐ ┌────▼───────────┐
│ Omnilingual ASR  │ │ Swahili CSM    │ │ Aya-101        │
│ (STT)            │ │ (TTS)          │ │ (LLM)          │
│                  │ │                │ │                │
│ - CTC-1B Model   │ │ - swa-csm-1b   │ │ - vLLM Server  │
│ - swh_Latn       │ │ - Multi-voice  │ │ - 13B params   │
│ - 16kHz input    │ │ - 24kHz output │ │ - Streaming    │
│ - WebSocket      │ │ - WebSocket    │ │ - HTTP/SSE     │
└──────────────────┘ └────────────────┘ └────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal setup
   ```

### Installation

1. **Navigate to the Swahili bot directory**:
   ```bash
   cd swahili_bot
   ```

2. **Deploy the services** (in separate terminals or sequentially):

   ```bash
   # Deploy STT service (Omnilingual ASR)
   modal deploy server/stt/omnilingual_stt.py

   # Deploy TTS service (Swahili CSM)
   modal deploy server/tts/swahili_csm_tts.py

   # Deploy LLM service (Aya-101)
   modal deploy server/llm/aya_vllm_server.py

   # Deploy the main app (frontend + bot)
   modal deploy app.py
   ```

3. **Access the application**:
   - After deploying `app.py`, Modal will provide a URL
   - Open the URL in your browser
   - Click "Anza Mazungumzo" (Start Conversation) to begin

## 📖 Usage Guide

### Basic Conversation

1. **Select Speaker ID**: Choose a voice by entering a speaker ID (0-100). Default is 22.
   - Try different values like 22, 25, 30 to find your preferred voice

2. **Start Conversation**: Click "Anza Mazungumzo" to connect

3. **Speak in Swahili**: The bot will transcribe your speech and respond naturally

4. **Mute/Unmute**: Use the "Zima Sauti" button to mute your microphone

5. **View Transcript**: See real-time transcription of your conversation

### Example Conversations

**Greeting:**
- You: "Habari yako Rafiki?"
- Rafiki: "Habari njema! Mimi ni msaidizi wako wa Kiswahili. Ninafuraha kukusaidia leo. Je, ungependa kujua chochote?"

**Asking about History:**
- You: "Eleza kidogo kuhusu historia ya Tanzania"
- Rafiki: [Provides explanation in Swahili]

**General Topics:**
- Education, science, culture, technology, health, and more!

## 🔧 Configuration

### Regional Settings

Edit `server/common/const.py` to change deployment regions:

```python
# Default: US West Coast (low latency from Bay Area)
SERVICE_REGIONS = ["us-west-1", "us-sanjose-1", "westus"]

# For East Coast:
# SERVICE_REGIONS = ["us-east-1", "us-ashburn-1", "us-east4"]

# For global deployment (may increase latency):
# SERVICE_REGIONS = None
```

### Model Parameters

**STT (Speech-to-Text):**
- Model: `omniASR_CTC_1B` (in `server/stt/omnilingual_stt.py`)
- Language: `swh_Latn` (Swahili)
- Sample Rate: 16kHz

**TTS (Text-to-Speech):**
- Model: `Nadhari/swa-csm-1b` (in `server/tts/swahili_csm_tts.py`)
- Sample Rate: 24kHz
- Max Tokens: 250 (~20 seconds of audio)
- Speaker ID: Configurable via UI (default: 22)

**LLM (Language Model):**
- Model: `CohereForAI/aya-101` (in `server/llm/aya_vllm_server.py`)
- Parameters: 13B
- Max Length: 2048 tokens
- Temperature: 0.7

### System Prompt

The bot's personality is defined in `server/bot/swahili_bot.py`:

```python
def get_swahili_system_prompt() -> str:
    return """Wewe ni msaidizi wa kirafiki na mwenye kueleweka..."""
```

You can customize this to change the bot's behavior, tone, and domain knowledge.

## 📊 Performance

### Latency Breakdown

Based on testing with client in Bay Area and services in `us-west`:

| Component | Latency | Notes |
|-----------|---------|-------|
| STT (Omnilingual CTC-1B) | ~50-100ms | 16-96x faster than real-time |
| LLM (Aya-101) | ~200-400ms | TTFT with vLLM optimization |
| TTS (swa-csm-1b) | ~200-300ms | Streaming synthesis |
| Network (WebRTC + Tunnels) | ~100-200ms | P2P with Modal Tunnels |
| **Total Voice-to-Voice** | **~550-1000ms** | Target: <1 second |

### GPU Requirements

- **STT**: A10G (cost-effective for 1B model)
- **TTS**: A10G (efficient for 1B model)
- **LLM**: H100 (required for 13B model with low latency)

### Cost Optimization

To reduce costs, you can:

1. **Use smaller LLM**: Consider using Aya-23-8B or smaller multilingual models
2. **Reduce context length**: Lower `max-model-len` in vLLM config
3. **Adjust idle timeouts**: Increase `container_idle_timeout` to reduce cold starts
4. **Enable snapshots**: All services use `enable_memory_snapshot=True` for faster cold starts

## 🛠️ Development

### Local Testing

Test individual services:

```bash
# Test STT service
modal run server/stt/omnilingual_stt.py

# Test TTS service
modal run server/tts/swahili_csm_tts.py

# Test LLM service
modal run server/llm/aya_vllm_server.py

# Test full app
modal serve app.py
```

### Frontend Development

```bash
cd client
npm install
npm run dev
```

This starts a local Vite dev server at `http://localhost:5173`

### Debugging

Enable debug logging by checking Modal logs:

```bash
modal app logs swahili-voice-bot
modal app logs swahili-omnilingual-transcription
modal app logs swahili-csm-tts
modal app logs swahili-aya-llm
```

## 🔍 Troubleshooting

### Common Issues

**1. "Failed to connect to bot"**
- Ensure all services are deployed and running
- Check Modal dashboard for service status
- Verify regional configuration matches

**2. "Slow response times"**
- Check client-to-service latency (try different regions)
- Monitor GPU utilization in Modal dashboard
- Consider using H100 instead of A100 for LLM

**3. "Poor transcription quality"**
- Ensure clear audio input
- Check microphone permissions in browser
- Verify language is Swahili (model is trained for `swh_Latn`)

**4. "Voice sounds different than expected"**
- Try different speaker IDs (0-100)
- Note: Not all speaker IDs may be available in the fine-tuned model

**5. "Service cold start is slow"**
- First request after idle timeout will be slower
- GPU snapshots help, but initial load takes time
- Consider keeping services warm with periodic pings

## 🔬 Model Information

### Omnilingual ASR
- **Source**: [Meta AI Research](https://github.com/facebookresearch/omnilingual-asr)
- **Model**: omniASR_CTC_1B
- **Languages**: 1600+ languages including Swahili
- **License**: Apache 2.0

### Swahili CSM-1B
- **Source**: [Nadhari/swa-csm-1b](https://huggingface.co/Nadhari/swa-csm-1b)
- **Base Model**: CSM-1B (Conversational Speech Model)
- **Fine-tune**: Swahili language
- **Sample Rate**: 24kHz

### Aya-101
- **Source**: [CohereForAI/aya-101](https://huggingface.co/CohereForAI/aya-101)
- **Parameters**: 13B (Seq2Seq architecture)
- **Languages**: 101 languages including Swahili
- **License**: Apache 2.0

## 📝 Project Structure

```
swahili_bot/
├── README.md                          # This file
├── app.py                             # Main Modal app (frontend + bot server)
├── server/
│   ├── common/
│   │   └── const.py                   # Regional configuration
│   ├── stt/
│   │   └── omnilingual_stt.py         # STT service
│   ├── tts/
│   │   └── swahili_csm_tts.py         # TTS service
│   ├── llm/
│   │   └── aya_vllm_server.py         # LLM service
│   └── bot/
│       ├── swahili_bot.py             # Main bot pipeline
│       └── services/
│           ├── modal_services.py      # Base service classes
│           ├── modal_omnilingual_service.py  # STT wrapper
│           ├── modal_swahili_tts_service.py  # TTS wrapper
│           └── modal_aya_service.py          # LLM wrapper
└── client/
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx
        ├── index.css
        └── components/
            ├── app.tsx                # Main React component
            └── app.css                # Styles
```

## 🤝 Contributing

Contributions are welcome! Some ideas:

- Add support for other Swahili TTS voices
- Implement conversation memory/history
- Add support for code-switching (Swahili-English)
- Optimize for mobile devices
- Add analytics and metrics
- Implement user authentication

## 📄 License

This project builds upon the original open-source-av-ragbot implementation.

Models used:
- Omnilingual ASR: Apache 2.0
- Aya-101: Apache 2.0
- Swahili CSM-1B: Check model card for license

## 🙏 Acknowledgments

- **Modal Team**: For the powerful serverless GPU platform
- **Pipecat**: For the excellent voice AI framework
- **Meta AI**: For Omnilingual ASR
- **Cohere for AI**: For Aya-101
- **Nadhari**: For the Swahili CSM fine-tune
- **Original Demo**: Based on [Modal's low-latency voice bot tutorial](https://modal.com/blog/low-latency-voice-bot)

## 📧 Support

For issues and questions:
- Open an issue in this repository
- Check Modal's [community Slack](https://modal.com/slack)
- Review [Modal documentation](https://modal.com/docs)

---

**Karibu! Welcome to Rafiki - Your Swahili AI Companion** 🎉
