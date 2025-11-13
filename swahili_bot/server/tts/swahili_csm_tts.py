"""
Swahili CSM-1B Text-to-Speech Service

This service uses the Nadhari/swa-csm-1b model, a Swahili fine-tune of the CSM-1B model.
It provides high-quality, streaming text-to-speech synthesis with multiple speaker voices.
"""

import modal
import asyncio
import json
import torch
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
import sys

# Import regional configuration
from swahili_bot.server.common.const import SERVICE_REGIONS

# Modal app configuration
app = modal.App("swahili-csm-tts")

# GPU image with all dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "soundfile==0.12.1",
        "fastapi[standard]==0.115.4",
        "numpy==1.26.4",
        "loguru==0.7.2",
    )
)

UVICORN_PORT = 8000
SAMPLE_RATE = 24000  # CSM models use 24kHz audio


@app.cls(
    image=gpu_image,
    gpu="A10G",  # A10G is sufficient for this 1B parameter model
    timeout=30 * 60,  # 30 minutes
    container_idle_timeout=5 * 60,  # 5 minutes idle timeout
    enable_memory_snapshot=True,
    regions=SERVICE_REGIONS,
)
class SwahiliTTS:
    @modal.enter(snap=True)
    def load_model(self):
        """Load the Swahili CSM-1B model on GPU startup."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("Loading Swahili CSM-1B TTS model...")

        model_id = "Nadhari/swa-csm-1b"

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use FP16 for faster inference
            device_map="cuda",
        )
        self.model.eval()

        logger.info("Model loaded successfully")

        # Warmup
        logger.info("Warming up model with sample text...")
        self._warmup()

    def _warmup(self, num_runs: int = 3):
        """Warmup the model with sample Swahili text."""
        import time

        sample_text = "Habari yako? Ninafuraha kukutana nawe leo."
        sample_speaker_id = 22

        latencies = []
        for i in range(num_runs):
            start = time.perf_counter()

            conversation = [
                {"role": str(sample_speaker_id), "content": [{"type": "text", "text": sample_text}]},
            ]

            with torch.no_grad():
                audio_values = self.model.generate(
                    **self.processor.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_dict=True,
                    ).to("cuda"),
                    max_new_tokens=125,  # 125 tokens = ~10 seconds of audio
                    output_audio=True,
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
        """Handle WebSocket connections for TTS."""
        await websocket.accept()
        logger.info("Client connected to Swahili TTS service")

        # Queues for async communication
        prompt_queue = asyncio.Queue()
        audio_queue = asyncio.Queue()

        async def recv_loop():
            """Receive text prompts from client."""
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "prompt":
                        text = message.get("text", "")
                        speaker_id = message.get("speaker_id", 22)  # Default speaker
                        max_tokens = message.get("max_tokens", 250)  # Default 250 tokens (~20s)

                        await prompt_queue.put({
                            "text": text,
                            "speaker_id": speaker_id,
                            "max_tokens": max_tokens,
                        })
                    elif message.get("type") == "end":
                        break
            except WebSocketDisconnect:
                logger.info("Client disconnected (recv_loop)")
            except Exception as e:
                logger.error(f"Error in recv_loop: {e}")
            finally:
                await prompt_queue.put(None)  # Signal end

        async def inference_loop():
            """Generate speech from text prompts."""
            try:
                while True:
                    prompt_data = await prompt_queue.get()

                    if prompt_data is None:
                        break

                    text = prompt_data["text"]
                    speaker_id = prompt_data["speaker_id"]
                    max_tokens = prompt_data["max_tokens"]

                    logger.info(f"Synthesizing: '{text}' (speaker_id={speaker_id})")

                    # Generate audio
                    audio_chunks = await self._generate_audio(text, speaker_id, max_tokens)

                    for chunk in audio_chunks:
                        await audio_queue.put(chunk)

            except Exception as e:
                logger.error(f"Error in inference_loop: {e}")
            finally:
                await audio_queue.put(None)  # Signal end

        async def send_loop():
            """Send audio chunks back to client."""
            try:
                while True:
                    audio_chunk = await audio_queue.get()

                    if audio_chunk is None:
                        break

                    # Send binary audio data (PCM16)
                    await websocket.send_bytes(audio_chunk)

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

    async def _generate_audio(self, text: str, speaker_id: int, max_tokens: int):
        """Generate audio from text using the CSM model."""
        try:
            # Prepare conversation format
            conversation = [
                {"role": str(speaker_id), "content": [{"type": "text", "text": text}]},
            ]

            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_values = await loop.run_in_executor(
                None,
                self._run_inference,
                conversation,
                max_tokens,
            )

            # Convert to PCM16 format
            audio = audio_values[0].to(torch.float32).cpu().numpy()

            # Convert to PCM16 (int16)
            pcm_data = (audio.clip(-1.0, 1.0) * 32767).astype(np.int16)

            # For streaming, we could chunk this, but for simplicity send as one chunk
            # In production, you might want to chunk for better streaming UX
            yield pcm_data.tobytes()

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return

    def _run_inference(self, conversation, max_tokens):
        """Run model inference (executed in thread pool)."""
        with torch.no_grad():
            audio_values = self.model.generate(
                **self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                ).to("cuda"),
                max_new_tokens=max_tokens,
                output_audio=True,
            )
        return audio_values

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
    """Test the TTS service locally."""
    tts = SwahiliTTS()
    print(f"WebSocket URL: {tts.websocket_url}")
    print("Service is running. Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping service...")
