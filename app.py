import asyncio
from pathlib import Path
from re import escape
import time

import modal

from server import SERVICE_REGIONS

APP_NAME = "modal-voice-assistant"

app = modal.App(APP_NAME)

# container specifications for the Pipecat pipeline
bot_image = (
    modal.Image.debian_slim(python_version="3.12",)
    .apt_install(
        "git",
        "ffmpeg",
    )
    .uv_pip_install(
        "pipecat-ai[webrtc,openai,silero,local-smart-turn,noisereduce,soundfile]==0.0.92",
        "chromadb",
        "sentence-transformers[openvino]",
        "llama-index==0.14.7",
        "llama-index-embeddings-openvino",
        "llama-index-vector-stores-chroma",
        "websocket-client",
        "aiofiles",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .add_local_dir("server", remote_path="/root/server")
)

bot_sessions_dict = modal.Dict.from_name(f"{APP_NAME}-bot-sessions", create_if_missing=True)
recordings_volume = modal.Volume.from_name(f"{APP_NAME}-recordings", create_if_missing=True)

MINUTES = 60  # seconds in a minute

with bot_image.imports():
    from loguru import logger
    from pipecat.transports.smallwebrtc.connection import (
        IceServer,
        SmallWebRTCConnection,
    )

    from server.bot.moe_and_dal_bot import run_bot

@app.cls(
    image=bot_image,
    timeout=30 * MINUTES,
    region=SERVICE_REGIONS,
    enable_memory_snapshot=True,
    max_inputs=1,
    volumes={"/recordings": recordings_volume},
    # min_containers=1,
)
class ModalVoiceAssistant:

    @modal.enter(snap=True)
    def load(self):
        from server.bot.processors.modal_rag import ChromaVectorDB
        self.chroma_db = ChromaVectorDB()

    @modal.method()
    async def run_bot(self, d: modal.Dict):
        """Launch the bot process with WebRTC connection and run the bot pipeline.

        Args:
            d (modal.Dict): A dictionary containing the WebRTC offer and ICE servers configuration.

        Raises:
            RuntimeError: If the bot pipeline fails to start or encounters an error.
        """
        

        try:
            start_time = time.time()
            offer = await d.get.aio("offer")
            ice_servers = await d.get.aio("ice_servers")
            ice_servers = [
                IceServer(
                    **ice_server,
                )
                for ice_server in ice_servers
            ]
            print(f"Time taken to get ice servers: {time.time() - start_time:.3f} seconds")
            webrtc_connection = SmallWebRTCConnection(ice_servers)
            await webrtc_connection.initialize(sdp=offer["sdp"], type=offer["type"])

            @webrtc_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                logger.info("WebRTC connection to bot closed.")

            print(f"Time taken to initialize WebRTC connection: {time.time() - start_time:.3f} seconds")
            start_time = time.time()

            print("Starting bot process.")
            bot_task = asyncio.create_task(
                run_bot(
                    webrtc_connection, 
                    self.chroma_db, 
                    enable_moe_and_dal=False
                )
            )

            print(f"Time taken to create bot task: {time.time() - start_time:.3f} seconds")
            start_time = time.time()

            answer = webrtc_connection.get_answer()
            await d.put.aio("answer", answer)

            print(f"Time taken to put answer: {time.time() - start_time:.3f} seconds")
            start_time = time.time()


            await bot_task


        except Exception as e:
            raise RuntimeError(f"Failed to start bot pipeline: {e}")

    @modal.method()
    def ping(self):
        return "pong"


frontend_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.115.12")
    .add_local_dir("server", remote_path="/root/server")
    .add_local_dir(
        Path(__file__).parent / "client/dist",
        remote_path="/frontend",
    )
)

with frontend_image.imports():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles

@app.function(image=frontend_image)
@modal.asgi_app()
@modal.concurrent(max_inputs=100)
def serve_frontend():
    """Create and configure the FastAPI application.

    This function initializes the FastAPI app with middleware, routes, and lifespan management.
    It is decorated to be used as a Modal ASGI app.
    """

    web_app = FastAPI()

    static_files = StaticFiles(directory="/frontend")
    web_app.mount("/frontend", static_files)

    @web_app.get("/")
    async def root():
        return FileResponse("/frontend/index.html")
    
    @web_app.post("/offer")
    async def offer(offer: dict):

        start_time = time.time()

        _ICE_SERVERS = [
            {
                "urls": "stun:stun.l.google.com:19302",
            }
        ]

        with modal.Dict.ephemeral() as d:

            await d.put.aio("ice_servers", _ICE_SERVERS)
            await d.put.aio("offer", offer)

            print(f"Time taken to put offer: {time.time() - start_time:.3f} seconds")
            start_time = time.time()

            bot_func_call = ModalVoiceAssistant().run_bot.spawn(d)

            print(f"Time taken to spawn bot: {time.time() - start_time:.3f} seconds")
            start_time = time.time()

            try:
                while True:
                    answer = await d.get.aio("answer")
                    if answer:
                        print(f"Time taken to get answer: {time.time() - start_time:.3f} seconds")
                        return answer
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error signaling with bot from server: {type(e)}: {e}")
                bot_func_call.cancel()
                raise e

    return web_app

# warm up snapshots if needed
if __name__ == "__main__":
    bot_server = modal.Cls.from_name(APP_NAME, "ModalVoiceAssistant")
    num_cold_starts = 50
    for _ in range(num_cold_starts):
        start_time = time.time()
        bot_server().ping.remote()
        end_time = time.time()
        print(f"Time taken to ping: {end_time - start_time:.3f} seconds")
        time.sleep(10.0) # allow container to drain
    print(f"ModalVoiceAssistant cold starts: {num_cold_starts}")