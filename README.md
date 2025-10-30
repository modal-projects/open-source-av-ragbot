# open-source-av-ragbot
Video chat with Modal's mascots, Moe and Dal, about Modal and its documentation.

# Dev Instructions for Daily

## Deploying STT, LLM, and TTS Services

I think the easiest way to use the app is for y'all to deploy the GPU services in your own workspaces. I could rework the code and provide URLs. But if a container restarts, the URL will change and this would block you since you couldn't retrieve the URL from a different workspace.

Run the bash script in the `utils` folder and provide one or more services to deploy as arguments:
```
./utils/ragbot_deploy.sh stt tts llm
```
The services are set to have `min_containers=1` so that you don't have to wait for cold start times if you've only redeployed the bot/frontend.

The expected cold-start times (once a container is provisioned) are
- Parakeet STT: ~ 60 sec
- vLLM Qwen3 Server: ~ 3 min
- Kokoro TTS: ~50 sec

## Deploying the bot
Use the same script to redeploy the bot:
```
./utils/ragbot_deploy.sh bot
```

The bot container takes about 20 seconds to cold start.

## Note on bot start up time
Warming up the RAG retriever takes about 15 seconds. I'd love to figure out how to optimize or snapshot this, but for now you'll have to wait about 20 seconds after you hit "Connect" in the browser for the bot to start.
