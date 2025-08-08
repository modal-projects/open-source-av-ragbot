# Built from Pipecat's Modal Deployment example here:
# https://github.com/pipecat-ai/pipecat/tree/main/examples/deployment/modal-example

import asyncio
from pathlib import Path

import modal

app = modal.App("moe-and-dal-ragbot")

# container specifications for the Pipecat pipeline
bot_image = (
    modal.Image.debian_slim(python_version="3.12",)
    .apt_install("ffmpeg")
    .pip_install(
        "pipecat-ai[webrtc,openai,silero,google,local-smart-turn,noisereduce]==0.0.76",
        "websocket-client",
        "aiofiles"
    )
    .add_local_dir("server", remote_path="/root/server")
)

MINUTES = 60  # seconds in a minute

ragbot_recordings_volume = modal.Volume.from_name("ragbot-recordings", create_if_missing=True)

@app.function(
    image=bot_image,
    gpu="L40S",
    timeout=30 * MINUTES,
    min_containers=1,
    # max_inputs=1,
    volumes={
        "/ragbot_recordings": ragbot_recordings_volume,
    },
)


async def run_bot(d: modal.Dict):
    """Launch the bot process with WebRTC connection and run the bot pipeline.

    Args:
        d (modal.Dict): A dictionary containing the WebRTC offer and ICE servers configuration.

    Raises:
        RuntimeError: If the bot pipeline fails to start or encounters an error.
    """
    from loguru import logger
    from pipecat.transports.network.webrtc_connection import (
        IceServer,
        SmallWebRTCConnection,
    )

    from .bot.moe_and_dal_bot import run_bot

    try:

        offer = await d.get.aio("offer")
        ice_servers = await d.get.aio("ice_servers")
        ice_servers = [
            IceServer(
                **ice_server,
            )
            for ice_server in ice_servers
        ]

        webrtc_connection = SmallWebRTCConnection(ice_servers)
        await webrtc_connection.initialize(sdp=offer["sdp"], type=offer["type"])

        @webrtc_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info("WebRTC connection to bot closed.")

        print("Starting bot process.")
        bot_task = asyncio.create_task(run_bot(webrtc_connection))

        answer = webrtc_connection.get_answer()
        await d.put.aio("answer", answer)

        await bot_task

    except Exception as e:
        raise RuntimeError(f"Failed to start bot pipeline: {e}")


web_server_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.115.12")
    .add_local_dir("server", remote_path="/root/server")
)


@app.function(
    image=web_server_image,
)
@modal.asgi_app()
def bot_server():
    """Create and configure the FastAPI application.

    This function initializes the FastAPI app with middleware, routes, and lifespan management.
    It is decorated to be used as a Modal ASGI app.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    # Initialize FastAPI app
    web_app = FastAPI()

    # # # setup CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/offer")
    async def offer(offer: dict):

        _ICE_SERVERS = [
            {
                "urls": "stun:stun.l.google.com:19302",
            }
        ]

        with modal.Dict.ephemeral() as d:

            await d.put.aio("ice_servers", _ICE_SERVERS)
            await d.put.aio("offer", offer)

            run_bot.spawn(d)

            while True:
                answer = await d.get.aio("answer")
                if answer:
                    return answer
                await asyncio.sleep(0.1)

    return web_app


frontend_image = web_server_image.add_local_dir(
    Path(__file__).parent.parent / "client/dist",
    remote_path="/frontend",
)


@app.function(image=frontend_image)
@modal.asgi_app()
def frontend_server():
    """Create and configure the FastAPI application.

    This function initializes the FastAPI app with middleware, routes, and lifespan management.
    It is decorated to be used as a Modal ASGI app.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    web_app = FastAPI()

    static_files = StaticFiles(directory="/frontend")
    web_app.mount("/frontend", static_files)

    @web_app.get("/")
    async def root():
        return HTMLResponse(content=open("/frontend/index.html").read())
    
    @web_app.post("/offer")
    async def offer(offer: dict):
        return RedirectResponse(url="https://modal-labs-shababo-dev--moe-and-dal-ragbot-bot-server.modal.run/offer")

    return web_app
