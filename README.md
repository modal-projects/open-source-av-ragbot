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

The bot container takes about 20 seconds to cold start, and we also set `min_containers=1` so the 20 seconds starts when you redeploy, not when you hit "Connect".

### Frontend URL
The front end URL will print to the console and look like:
```
├── 🔨 Created function serve_bot.
└── 🔨 Created web function serve_frontend => https://{workspace}-{environment}--moe-and-dal-ragbot-serve-frontend.modal.run
```

You can also find this URL in the web function's (`serve_frontend`) view in your Modal Dashboard.

## Note on bot start up time
Warming up the RAG retriever takes about 15 seconds. I'd love to figure out how to optimize or snapshot this, but for now you'll have to wait about 20 seconds after you hit "Connect" in the browser for the bot to start.
