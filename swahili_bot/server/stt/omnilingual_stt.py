"""
Omnilingual ASR Speech-to-Text Service for Swahili

This service uses Meta's Omnilingual ASR CTC-1B model for fast, accurate Swahili transcription.
The CTC variant provides 16-96x faster-than-real-time inference, ideal for low-latency applications.
"""

import modal
import asyncio
import base64
import json
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
import torch
import sys
import io
import soundfile as sf

# Import regional configuration
from swahili_bot.server.common.const import SERVICE_REGIONS

# Modal app configuration
app = modal.App("swahili-omnilingual-transcription")

# GPU image with all dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "omnilingual-asr==0.1.0",
        "torch==2.5.1",
        "fastapi[standard]==0.115.4",
        "soundfile==0.12.1",
        "numpy==1.26.4",
        "loguru==0.7.2",
    )
)

UVICORN_PORT = 8000
SAMPLE_RATE = 16000


@app.cls(
    image=gpu_image,
    gpu="A10G",  # A10G is cost-effective for this smaller model
    timeout=30 * 60,  # 30 minutes
    container_idle_timeout=5 * 60,  # 5 minutes idle timeout
    enable_memory_snapshot=True,
    regions=SERVICE_REGIONS,
)
class SwahiliTranscriber:
    @modal.enter(snap=True)
    def load_model(self):
        """Load the Omnilingual ASR CTC-1B model on GPU startup."""
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        logger.info("Loading Omnilingual ASR CTC-1B model...")

        # Use CTC-1B for speed (16-96x faster than real-time)
        self.pipeline = ASRInferencePipeline(
            model_card="omniASR_CTC_1B",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        logger.info("Model loaded successfully")

        # Warmup
        logger.info("Warming up model with sample audio...")
        self._warmup()

    def _warmup(self, num_runs: int = 3):
        """Warmup the model with dummy audio."""
        import time

        # Create dummy audio (1 second of silence at 16kHz)
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        latencies = []
        for i in range(num_runs):
            start = time.perf_counter()
            _ = self.pipeline.transcribe(
                [{"waveform": dummy_audio, "sample_rate": SAMPLE_RATE}],
                lang=["swh_Latn"],  # Swahili language code
                batch_size=1,
            )
            latency = time.perf_counter() - start
            latencies.append(latency)
            logger.info(f"Warmup run {i+1}/{num_runs}: {latency:.3f}s")

        logger.info(
            f"Warmup complete. Avg latency: {np.mean(latencies):.3f}s "
            f"(p50: {np.median(latencies):.3f}s)"
        )

    @modal.enter(snap=False)
    async def start_server(self):
        """Start the WebSocket server using Modal Tunnel."""
        from fastapi import FastAPI
        import uvicorn

        # Create FastAPI app
        self.fastapi_app = FastAPI()

        @self.fastapi_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)

        # Create Modal Tunnel
        self.tunnel_ctx = modal.forward(UVICORN_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"

        logger.info(f"WebSocket URL: {self.websocket_url}")

        # Start uvicorn server in background
        config = uvicorn.Config(
            self.fastapi_app,
            host="0.0.0.0",
            port=UVICORN_PORT,
            log_level="info",
        )
        self.server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(self.server.serve())

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections for audio transcription."""
        await websocket.accept()
        logger.info("Client connected to Swahili transcription service")

        # Queues for async communication between loops
        audio_queue = asyncio.Queue()
        transcription_queue = asyncio.Queue()

        async def recv_loop():
            """Receive audio data from client."""
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "audio":
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(message["audio"])
                        await audio_queue.put(audio_bytes)
                    elif message.get("type") == "end":
                        break
            except WebSocketDisconnect:
                logger.info("Client disconnected (recv_loop)")
            except Exception as e:
                logger.error(f"Error in recv_loop: {e}")
            finally:
                await audio_queue.put(None)  # Signal end

        async def inference_loop():
            """Process audio and generate transcriptions."""
            audio_buffer = bytearray()

            try:
                while True:
                    audio_chunk = await audio_queue.get()

                    if audio_chunk is None:
                        # Process any remaining audio
                        if len(audio_buffer) > 0:
                            transcript = await self._transcribe_audio(bytes(audio_buffer))
                            if transcript:
                                await transcription_queue.put(transcript)
                        break

                    # Add to buffer
                    audio_buffer.extend(audio_chunk)

                    # Transcribe when we have enough audio (e.g., 0.5 seconds)
                    # This creates a segmented STT experience
                    min_samples = SAMPLE_RATE // 2  # 0.5 seconds
                    if len(audio_buffer) >= min_samples * 2:  # 2 bytes per int16 sample
                        transcript = await self._transcribe_audio(bytes(audio_buffer))
                        if transcript:
                            await transcription_queue.put(transcript)
                        audio_buffer.clear()

            except Exception as e:
                logger.error(f"Error in inference_loop: {e}")
            finally:
                await transcription_queue.put(None)  # Signal end

        async def send_loop():
            """Send transcriptions back to client."""
            try:
                while True:
                    transcript = await transcription_queue.get()

                    if transcript is None:
                        break

                    # Send as plain text (matching Parakeet service format)
                    await websocket.send_text(transcript)

            except WebSocketDisconnect:
                logger.info("Client disconnected (send_loop)")
            except Exception as e:
                logger.error(f"Error in send_loop: {e}")

        # Run all loops concurrently
        await asyncio.gather(
            recv_loop(),
            inference_loop(),
            send_loop(),
            return_exceptions=True,
        )

        logger.info("WebSocket handler completed")

    async def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        try:
            # Convert bytes to numpy array
            audio_io = io.BytesIO(audio_bytes)
            audio_data, sr = sf.read(audio_io, dtype="float32")

            # Ensure mono channel
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample if needed (though client should send 16kHz)
            if sr != SAMPLE_RATE:
                logger.warning(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)

            # Skip very short audio
            if len(audio_data) < SAMPLE_RATE * 0.3:  # Less than 0.3 seconds
                return ""

            # Transcribe with Swahili language code
            # Note: Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            transcriptions = await loop.run_in_executor(
                None,
                lambda: self.pipeline.transcribe(
                    [{"waveform": audio_data, "sample_rate": SAMPLE_RATE}],
                    lang=["swh_Latn"],  # Swahili language code
                    batch_size=1,
                ),
            )

            transcript = transcriptions[0].strip()
            if transcript:
                logger.info(f"Transcription: {transcript}")

            return transcript

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""

    @modal.method()
    async def run_tunnel_client(self, url_dict: modal.Dict):
        """Method to expose the WebSocket URL via Modal Dict."""
        await url_dict.put.aio("url", self.websocket_url)
        logger.info(f"Published WebSocket URL: {self.websocket_url}")

        # Keep alive until cancelled
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Tunnel client cancelled")


@app.local_entrypoint()
def main():
    """Test the transcription service locally."""
    transcriber = SwahiliTranscriber()
    print(f"WebSocket URL: {transcriber.websocket_url}")
    print("Service is running. Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping service...")
