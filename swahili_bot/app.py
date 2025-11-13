"""
Swahili Voice AI Chatbot - Main Application

This Modal app serves the frontend and manages bot sessions.
"""

import modal
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json

from swahili_bot.server.common.const import SERVICE_REGIONS

# Define Modal app
app = modal.App("swahili-voice-bot")

# Frontend image with Node.js for building React app
frontend_image = (
    modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
    )
    .copy_local_dir("client", "/assets/client")
    .workdir("/assets/client")
    .run_commands(
        "npm install",
        "npm run build",
    )
)

# Bot image with Pipecat and dependencies
bot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "pipecat-ai[webrtc,openai,silero,local-smart-turn,noisereduce,soundfile]==0.0.92",
        "fastapi[standard]==0.115.4",
        "loguru==0.7.2",
        "openai==1.57.2",
    )
)

MINUTES = 60


@app.function(image=frontend_image)
@modal.asgi_app()
def serve_frontend():
    """Serve the React frontend and handle WebRTC connections."""
    web_app = FastAPI()

    # Serve static files (built React app)
    dist_path = Path("/assets/client/dist")
    web_app.mount("/assets", StaticFiles(directory=dist_path / "assets"), name="assets")

    @web_app.get("/")
    async def read_root():
        return FileResponse(dist_path / "index.html")

    @web_app.post("/offer")
    async def handle_offer(request: dict):
        """
        Handle WebRTC offer from client and spawn bot session.

        The client sends an SDP offer with optional speaker_id parameter.
        We spawn a bot session and return the SDP answer.
        """
        sdp = request.get("sdp")
        speaker_id = request.get("speaker_id", 22)  # Default speaker ID

        if not sdp:
            return JSONResponse({"error": "No SDP provided"}, status_code=400)

        # Create connection data
        connection_data = modal.Dict.from_name(
            f"webrtc-connection",
            create_if_missing=True,
        )

        # Store the offer
        await connection_data.put.aio("offer", sdp)
        await connection_data.put.aio("speaker_id", speaker_id)

        # Spawn bot server
        bot_server = BotServer()
        await bot_server.serve_bot.remote.aio(connection_data)

        # Get the answer
        answer = await connection_data.get.aio("answer")

        return JSONResponse({"sdp": answer})

    return web_app


@app.cls(
    image=bot_image,
    timeout=30 * MINUTES,
    enable_memory_snapshot=True,
    regions=SERVICE_REGIONS,
)
class BotServer:
    """Bot server that manages individual conversation sessions."""

    @modal.method()
    async def serve_bot(self, connection_data: modal.Dict):
        """
        Serve a bot session for a WebRTC connection.

        Args:
            connection_data: Modal Dict containing connection parameters
        """
        from swahili_bot.server.bot.swahili_bot import run_bot
        from loguru import logger

        # Get connection parameters
        offer = await connection_data.get.aio("offer")
        speaker_id = await connection_data.get.aio("speaker_id", 22)

        logger.info(f"Starting bot session with speaker_id={speaker_id}")

        # Create WebRTC connection object
        webrtc_connection = type(
            "WebRTCConnection",
            (),
            {
                "offer": offer,
                "answer_dict": connection_data,
                "answer_key": "answer",
            },
        )()

        # Run the bot
        await run_bot(webrtc_connection, speaker_id=speaker_id)

        logger.info("Bot session ended")


@app.local_entrypoint()
def main():
    """Deploy and test the application."""
    print("Swahili Voice AI Chatbot")
    print("=" * 50)
    print("\nDeploying application...")
    print("\nOnce deployed, visit the URL shown above to interact with the bot.")
    print("\nYou can:")
    print("- Speak in Swahili and the bot will respond")
    print("- Adjust the speaker ID to change the voice")
    print("- Have natural conversations on any topic")
