# Built from Pipecat's Modal deployment example
# https://github.com/pipecat-ai/pipecat/tree/main/examples/deployment/modal-example


import asyncio
import modal



# container specifications for the FastAPI web server
web_server_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.115.12")
    .add_local_dir("server", remote_path="/root/server")
)

# container specifications for the Pipecat pipeline
bot_image = (
    modal.Image.debian_slim(python_version="3.12",)
    .apt_install("ffmpeg")
    .pip_install(
        "pipecat-ai[webrtc,openai,silero,google,local-smart-turn]", 
        "loguru"
    )
    .add_local_dir("server", remote_path="/root/server")
)

app = modal.App("moe-and-dal-ragbot")

container_addresses = modal.Dict.from_name("ragbot_container_address", create_if_missing=True)

@app.function(image=bot_image, min_containers=1, gpu="L40S")
async def bot_runner(d: modal.Dict):
    """Launch the provided bot process, providing the given room URL and token for the bot to join.

    Args:
        bot_name (BotName): The name of the bot implementation to use. Defaults to "openai".

    Raises:
        HTTPException: If the bot pipeline fails to start.
    """
    from loguru import logger
    from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection, IceServer

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

        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=offer["sdp"], type=offer["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection")
            pipecat_connection = None            

        print(f"Starting bot process...")
        
        answer = pipecat_connection.get_answer()
        await d.put.aio("answer", answer)

        bot_task = asyncio.create_task(run_bot(pipecat_connection))
        await bot_task

    except Exception as e:
        raise RuntimeError(f"Failed to start bot pipeline: {e}")

    

@app.function(image=web_server_image, min_containers=1)
@modal.concurrent(max_inputs=1)
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

        ice_servers = [
            {
                "urls": "stun:stun.l.google.com:19302",
            }
        ]

        with modal.Dict.ephemeral() as d:

            await d.put.aio("ice_servers", ice_servers)
            await d.put.aio("offer", offer)

            bot_runner.spawn(d)

            while True: 
                answer = await d.get.aio("answer")
                if answer:
                    return answer
                await asyncio.sleep(0.1)

    return web_app

from pathlib import Path

frontend_image = web_server_image.add_local_dir(  
        Path(__file__).parent.parent / "client/dist",
        remote_path="/frontend",
    )
@app.function(image=frontend_image, min_containers=1)
@modal.asgi_app()
def frontend_server():
    """Create and configure the FastAPI application.

    This function initializes the FastAPI app with middleware, routes, and lifespan management.
    It is decorated to be used as a Modal ASGI app.
    """
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    
    web_app = FastAPI()

    static_files = StaticFiles(directory="/frontend")
    web_app.mount("/frontend", static_files)

    @web_app.get("/")
    async def root():
        return HTMLResponse(content=open("/frontend/index.html").read())

    return web_app