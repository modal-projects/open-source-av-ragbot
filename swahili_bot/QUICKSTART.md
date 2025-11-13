# ⚡ Quick Start Guide

Get your Swahili Voice AI bot running in 5 minutes!

## Step 1: Install Modal (1 minute)

```bash
pip install modal
modal setup
```

Follow the prompts to authenticate with your Modal account.

## Step 2: Deploy Services (3 minutes)

From the `swahili_bot` directory, run:

```bash
# Deploy all services
modal deploy server/stt/omnilingual_stt.py
modal deploy server/tts/swahili_csm_tts.py
modal deploy server/llm/aya_vllm_server.py
modal deploy app.py
```

**Note:** First deployment downloads models and may take 5-10 minutes total. Subsequent deployments are much faster thanks to caching!

## Step 3: Open the App (30 seconds)

After deploying `app.py`, you'll see a URL like:
```
https://your-username--swahili-voice-bot-serve-frontend.modal.run
```

Open this URL in Chrome or Firefox.

## Step 4: Start Talking! (instantly)

1. Click "Anza Mazungumzo" (Start Conversation)
2. Allow microphone access when prompted
3. Speak in Swahili!

Try saying:
- "Habari yako?" (How are you?)
- "Eleza kuhusu Tanzania" (Tell me about Tanzania)
- "Nini maana ya uhuru?" (What is the meaning of freedom?)

## Troubleshooting

**Can't hear the bot?**
- Check your browser's sound settings
- Make sure you allowed audio playback

**Bot doesn't understand you?**
- Speak clearly in Swahili
- Check microphone permissions
- Ensure you're not muted

**Slow responses?**
- First request after idle may be slower (cold start)
- Try refreshing and reconnecting

## Next Steps

- Try different speaker IDs (change the number in the UI)
- Read the full [README.md](README.md) for customization options
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for advanced deployment tips

## Getting Help

- Check Modal logs: `modal app logs swahili-voice-bot`
- View service status in [Modal Dashboard](https://modal.com/apps)
- Review error messages in browser console (F12)

---

Enjoy your Swahili AI conversation! 🎉
