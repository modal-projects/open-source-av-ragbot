# 🚀 Deployment Guide

Quick reference for deploying the Swahili Voice AI Bot to Modal.

## Prerequisites

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup
```

## Step-by-Step Deployment

### 1. Deploy Services (One-time setup)

Run these commands from the `swahili_bot` directory:

```bash
# Deploy STT service (Omnilingual ASR)
modal deploy server/stt/omnilingual_stt.py

# Deploy TTS service (Swahili CSM)
modal deploy server/tts/swahili_csm_tts.py

# Deploy LLM service (Aya-101)
modal deploy server/llm/aya_vllm_server.py
```

**Expected output:**
- Each service will create a Modal app
- GPU containers will be provisioned
- Models will be downloaded and cached
- Services will be available at generated URLs

**Initial deployment time:**
- STT: ~2-3 minutes
- TTS: ~2-3 minutes
- LLM: ~5-7 minutes (larger model)

### 2. Deploy Main Application

```bash
# Deploy the frontend and bot orchestrator
modal deploy app.py
```

**Expected output:**
- Frontend will be built (npm install + build)
- Bot server will be configured
- You'll receive a public URL like: `https://your-username--swahili-voice-bot-serve-frontend.modal.run`

### 3. Access the Application

Open the URL from step 2 in your browser. You should see:

- 🎤 Rafiki interface
- Speaker ID input field (default: 22)
- "Anza Mazungumzo" button to start

## Testing Individual Services

### Test STT Service

```bash
modal run server/stt/omnilingual_stt.py
```

This will:
- Start the Omnilingual ASR service
- Print the WebSocket URL
- Keep the service running until you press Ctrl+C

### Test TTS Service

```bash
modal run server/tts/swahili_csm_tts.py
```

This will:
- Start the Swahili CSM TTS service
- Print the WebSocket URL
- Keep the service running

### Test LLM Service

```bash
modal run server/llm/aya_vllm_server.py
```

This will:
- Start the Aya-101 vLLM server
- Run a test prompt: "Eleza kwa kifupi historia ya Tanzania"
- Print the response
- Display the API URL

### Test Full Application (Development Mode)

```bash
modal serve app.py
```

This runs the app in development mode with:
- Live code reloading
- Local logs visible in terminal
- Same functionality as deployed version

## Monitoring Deployments

### View Running Apps

```bash
modal app list
```

### View App Logs

```bash
# Main app logs
modal app logs swahili-voice-bot

# STT service logs
modal app logs swahili-omnilingual-transcription

# TTS service logs
modal app logs swahili-csm-tts

# LLM service logs
modal app logs swahili-aya-llm
```

### Monitor GPU Usage

Visit the [Modal Dashboard](https://modal.com/apps) to:
- See GPU utilization
- Monitor costs
- View request metrics
- Check container status

## Updating Services

### Update Service Code

After making changes:

```bash
# Redeploy specific service
modal deploy server/stt/omnilingual_stt.py

# Or redeploy all
modal deploy server/stt/omnilingual_stt.py
modal deploy server/tts/swahili_csm_tts.py
modal deploy server/llm/aya_vllm_server.py
modal deploy app.py
```

### Update Frontend Only

```bash
# Just redeploy the main app (includes frontend rebuild)
modal deploy app.py
```

## Regional Configuration

To change deployment regions, edit `server/common/const.py`:

```python
# For US West Coast (default)
SERVICE_REGIONS = ["us-west-1", "us-sanjose-1", "westus"]

# For US East Coast
SERVICE_REGIONS = ["us-east-1", "us-ashburn-1", "us-east4"]

# For global (may increase latency)
SERVICE_REGIONS = None
```

Then redeploy all services.

## Troubleshooting Deployment

### Issue: "App not found"

**Solution:**
Deploy the service first:
```bash
modal deploy server/stt/omnilingual_stt.py
```

### Issue: "GPU not available"

**Solution:**
- Try different regions (edit `const.py`)
- Use different GPU types (edit service files)
- Wait a few minutes and retry

### Issue: "Build failed"

**Solution:**
- Check your internet connection
- Verify Modal CLI is up to date: `pip install --upgrade modal`
- Check Modal status: https://status.modal.com

### Issue: "Frontend won't load"

**Solution:**
- Clear browser cache
- Check browser console for errors
- Verify `app.py` deployed successfully
- Check that all services are running

### Issue: "High latency"

**Solution:**
1. Check client region matches service region
2. Use Modal Dashboard to monitor service response times
3. Consider using faster GPUs (H100 > A100 > A10G)
4. Reduce `max-model-len` in LLM config

## Cost Management

### Estimated Costs (as of 2024)

Per hour of active usage:

- **STT (A10G)**: ~$0.60/hour
- **TTS (A10G)**: ~$0.60/hour
- **LLM (H100)**: ~$4.50/hour
- **Bot Server (CPU)**: ~$0.10/hour

**Total**: ~$5.80/hour when all services are active

**Idle costs**: Services scale to zero when not in use!

### Optimization Tips

1. **Increase idle timeout** (slower cold starts, less cost):
   ```python
   container_idle_timeout=10 * 60  # 10 minutes
   ```

2. **Use smaller models**:
   - Replace Aya-101 with Aya-23-8B
   - Use Omnilingual ASR 300M instead of 1B

3. **Enable snapshots** (already enabled):
   ```python
   enable_memory_snapshot=True
   ```

4. **Batch requests** if building a production service

## Production Checklist

Before going to production:

- [ ] Test with multiple concurrent users
- [ ] Set up monitoring and alerting
- [ ] Configure proper error handling
- [ ] Add rate limiting if needed
- [ ] Set appropriate timeouts
- [ ] Review and optimize costs
- [ ] Add authentication if needed
- [ ] Test in target deployment region
- [ ] Set up CI/CD for updates
- [ ] Document any customizations

## Quick Commands Reference

```bash
# Deploy everything
modal deploy server/stt/omnilingual_stt.py && \
modal deploy server/tts/swahili_csm_tts.py && \
modal deploy server/llm/aya_vllm_server.py && \
modal deploy app.py

# View all logs
modal app logs swahili-voice-bot --follow

# Stop all services (if running in development mode)
# Ctrl+C in terminal, or:
modal app stop swahili-voice-bot
modal app stop swahili-omnilingual-transcription
modal app stop swahili-csm-tts
modal app stop swahili-aya-llm
```

## Next Steps

After successful deployment:

1. **Test the application** with various Swahili prompts
2. **Monitor performance** via Modal Dashboard
3. **Optimize costs** by adjusting timeouts and GPU types
4. **Customize** the system prompt and voice settings
5. **Share** the URL with users

---

Happy deploying! 🚀

For issues, check the main [README.md](README.md) or Modal's [documentation](https://modal.com/docs).
