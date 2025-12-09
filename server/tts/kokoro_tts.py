import asyncio
import time
import json

import modal

from server import SERVICE_REGIONS, TTS_GPU

def chunk_audio(audio, desired_frame_size):
    for i in range(0, len(audio), desired_frame_size):
        yield audio[i:i + desired_frame_size]
    if len(audio) % desired_frame_size != 0:
        yield audio[-(len(audio) % desired_frame_size):]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "kokoro==0.9.4",
        # "soundfile",
        "fastapi[standard]",
        "pydub",
        "uvicorn[standard]",
    )
    .env({
        "HF_HOME": "/cache",
    })
)
app = modal.App("kokoro-tts")

with image.imports():
    from kokoro import KPipeline, KModel
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
    from pydub import AudioSegment
    import threading
    import uvicorn
    from server.common.model_pool import ModelPool
    import torch

DEFAULT_VOICE = 'am_puck'
UVICORN_PORT = 8000
MAX_CONCURRENT = 10
kokoro_hf_cache = modal.Volume.from_name("kokoro-tts-volume", create_if_missing=True)

@app.cls(
    image=image,
    volumes={"/cache": kokoro_hf_cache},
    gpu=TTS_GPU,
    # NOTE, uncomment min_containers = 1 for testing and avoiding cold start times
    min_containers=1, 
    region=SERVICE_REGIONS,
    timeout= 60 * 60,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=10,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class KokoroTTS:

    @modal.enter(snap=True)
    async def load(self):
        
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None

        self.model_pool = ModelPool()

        # Create model pool with CUDA streams for GPU parallelism
        for _ in range(MAX_CONCURRENT):
            model = KModel().to("cuda").eval()
            pipeline = KPipeline(model=model, lang_code='a', device="cuda")
            stream = torch.cuda.Stream()
            # Store (pipeline, stream) tuple
            await self.model_pool.put((pipeline, stream))

        for _ in range(MAX_CONCURRENT):
            async with self.model_pool.acquire_model() as (pipeline, stream):
                print("🔥 Warming up the model...")
                warmup_runs = 6
                warm_up_prompt = "Hello, we are Moe and Dal, your guides to Modal. We can help you get started with Modal, a platform that lets you run your Python code in the cloud without worrying about the infrastructure. We can walk you through setting up an account, installing the package, and running your first job."
            
                for _ in range(warmup_runs):
                    async for _ in self._stream_tts(pipeline, stream, warm_up_prompt):
                        pass

        print("✅ Models warmed up!")

    @modal.enter(snap=False)
    async def restore(self):

        self.webapp = FastAPI()

        @self.webapp.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):

            prompt_queue = asyncio.Queue()
            audio_queue = asyncio.Queue()


            async def recv_loop(ws, prompt_queue):
                while True:
                    msg = await ws.receive_text()
                    try:
                        json_data = json.loads(msg)
                        if "type" in json_data:
                            if json_data["type"] == "prompt":
                                print(f"Received prompt: {json_data['text']} with voice {json_data['voice']}")
                                await prompt_queue.put(json_data)
                                
                            else:
                                continue
                        else:
                            continue
                    except Exception as e:
                        continue
                    
                    
            async def inference_loop(prompt_queue, audio_queue):
                while True:
                    try:
                        prompt_msg = await prompt_queue.get()
                        print(f"Received prompt msg: {prompt_msg}")
                        start_time = time.perf_counter()
                        async with self.model_pool.acquire_model() as (pipeline, stream):
                            async for chunk in self._stream_tts(pipeline, stream, prompt_msg['text'], voice=prompt_msg['voice'], speed=prompt_msg['speed']):
                                print(f"Sending audio data to queue: {len(chunk)} bytes")
                                await audio_queue.put(chunk)
                        end_time = time.perf_counter()
                        print(f"Time taken to stream TTS: {end_time - start_time:.3f} seconds")

                    except Exception as e:
                        continue

                        
            async def send_loop(audio_queue, ws):
                while True:
                    audio = await audio_queue.get()
                    
                    await ws.send_bytes(audio)
                    print(f"sending audio data: {len(audio)} bytes")

            await ws.accept()

            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, prompt_queue)),
                    asyncio.create_task(inference_loop(prompt_queue, audio_queue)),
                    asyncio.create_task(send_loop(audio_queue, ws)),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                ws = None
            except Exception as e:
                print("Exception:", e)
            finally:
                if ws and ws.application_state is WebSocketState.CONNECTED:
                    await ws.close(code=1011) # internal error
                    ws = None
                for task in tasks:                    
                    if not task.done():
                        try:
                            task.cancel()
                            await task
                        except asyncio.CancelledError:
                            pass

                

        def start_server():
            uvicorn.run(self.webapp, host="0.0.0.0", port=UVICORN_PORT)

        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.tunnel_ctx = modal.forward(UVICORN_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"
        print(f"Websocket URL: {self.websocket_url}")
    
    @modal.asgi_app()
    def web_endpoint(self):
        
        return self.webapp

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        try:
            print(f"Sending websocket url: {self.websocket_url}")
            await d.put.aio("url", self.websocket_url)
            
            while not await d.contains.aio("is_running"):
                await asyncio.sleep(1.0)

            print("Tunnel client is running. Waiting for it to finish.")

            while await d.get.aio("is_running"):
                await asyncio.sleep(1.0)

            print("Tunnel client finished.")

        except Exception as e:
            print(f"Error running tunnel client: {type(e)}: {e}")

    @modal.method()
    def ping(self):
        return "pong"

    @modal.exit()
    def exit(self):
        if self.tunnel_ctx:
            self.tunnel_ctx.__exit__()
            self.tunnel_ctx = None
            self.tunnel = None
            self.websocket_url = None
    
    def _chunk_to_numpy(self, chunk):
        """
        Converts GPU tensor chunk to numpy array.
        This runs in a thread pool worker to avoid blocking the event loop.
        """
        # Ensure tensor is on CPU and convert to numpy for efficiency
        audio_numpy = chunk.to("cpu", non_blocking=True).numpy()

        audio_numpy = audio_numpy.clip(-1.0, 1.0) * 32767
        audio_numpy = audio_numpy.astype('int16')
        
        return audio_numpy
    async def _stream_tts(self, pipeline, stream, prompt: str, voice = None, speed = 1.3):

        if voice is None:
            voice = DEFAULT_VOICE

        try:
            stream_start = time.perf_counter()
            chunk_count = 0
            first_chunk_time = None

            # Generate streaming audio from the input text
            print(f"🎤 Starting streaming generation for prompt: {prompt}")
            
            # Create a queue for producer-consumer pattern
            # This allows generator to run in worker thread while we consume in event loop
            import queue
            chunk_queue = queue.Queue(maxsize=3)  # Buffer a few chunks
            
            # Start generator in worker thread
            # CRITICAL: Iterator must be created in worker thread with stream context!
            def _generate_in_worker():
                try:
                    with torch.cuda.stream(stream):
                        # Create iterator HERE in worker thread, not in main thread
                        for (gs, ps, chunk) in pipeline(
                            prompt,
                            voice=voice,
                            speed=speed,
                        ):
                            if chunk is None:
                                continue
                            
                            # Convert audio tensor to numpy (blocking operation)
                            audio_numpy = self._chunk_to_numpy(chunk)
                            chunk_queue.put((gs, ps, audio_numpy))
                    
                    # Signal completion
                    chunk_queue.put(None)
                except Exception as e:
                    print(f"Error in generation worker: {e}")
                    chunk_queue.put(None)
            
            # Start worker thread (asyncio.to_thread returns immediately)
            import threading
            worker = threading.Thread(target=_generate_in_worker, daemon=True)
            worker.start()
            
            # Stream chunks by consuming from queue
            while True:
                # Get next chunk from queue (blocks if empty, but in thread pool)
                chunk_data = await asyncio.to_thread(chunk_queue.get)
                
                if chunk_data is None:
                    break  # Generator finished
                
                gs, ps, audio_numpy = chunk_data
                
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    print(f"⏱️  Time to first chunk: {(first_chunk_time - stream_start):.3f} seconds")
                
                chunk_count += 1
                if chunk_count % 10 == 0:  # Log every 10th chunk
                    print(f"📊 Streamed {chunk_count} chunks so far")
                
                try:
                    # Fast post-processing in main thread (event loop)
                    audio_segment = AudioSegment(
                        audio_numpy.tobytes(),
                        frame_rate=24000,
                        sample_width=2,
                        channels=1
                    )

                    def detect_leading_silence(sound, silence_threshold=-60.0, chunk_size=10):
                        trim_ms = 0  # ms
                        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
                            trim_ms += chunk_size

                        return trim_ms - chunk_size # return the index of the last chunk with silence for padding

                    speech_start_idx = detect_leading_silence(audio_segment)
                    audio_segment = audio_segment[speech_start_idx:]
                    yield audio_segment.raw_data
                    
                except Exception as e:
                    print(f"❌ Error processing chunk {chunk_count}: {e}")
                    print(f"   Audio numpy shape: {audio_numpy.shape if hasattr(audio_numpy, 'shape') else 'N/A'}")
                    continue  # Skip this chunk and continue
            
            final_time = time.perf_counter()
            print(f"⏱️  Total streaming time: {final_time - stream_start:.3f} seconds")
            print(f"📊 Total chunks streamed: {chunk_count}")
            print("✅ KokoroTTS streaming complete!")

            
        except Exception as e:
            print(f"❌ Error creating stream generator: {e}")
            raise


def get_kokoro_server_url():
    try:
        return KokoroTTS().web_endpoint.get_web_url()
    except Exception as e:
        print(f"❌ Error getting Kokoro server URL: {e}")
        return None

# warm up snapshots if needed
if __name__ == "__main__":
    kokoro_tts = modal.Cls.from_name("kokoro-tts", "KokoroTTS")
    num_cold_starts = 50
    for _ in range(num_cold_starts):
        start_time = time.time()
        kokoro_tts().ping.remote()
        end_time = time.time()
        print(f"Time taken to ping: {end_time - start_time:.3f} seconds")
        time.sleep(20.0) # allow container to drain
    print(f"Kokoro TTS cold starts: {num_cold_starts}")
