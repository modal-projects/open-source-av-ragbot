import asyncio
import time
import json

import modal

from server import SERVICE_REGIONS

def chunk_audio(audio, desired_frame_size):
    for i in range(0, len(audio), desired_frame_size):
        yield audio[i:i + desired_frame_size]
    if len(audio) % desired_frame_size != 0:
        yield audio[-(len(audio) % desired_frame_size):]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "kokoro>=0.9.4",
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

DEFAULT_VOICE = 'am_puck'
UVICORN_PORT = 8000

kokoro_hf_cache = modal.Volume.from_name("kokoro-tts-volume", create_if_missing=True)

@app.cls(
    image=image,
    volumes={"/cache": kokoro_hf_cache},
    gpu="L40S",
    # NOTE, uncomment min_containers = 1 for testing and avoiding cold start times
    # min_containers=1, 
    region=SERVICE_REGIONS,
    timeout= 60 * 60,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=10)
class KokoroTTS:

    @modal.enter(snap=True)
    async def load(self):
        
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None
        
        self.model = KModel().to("cuda").eval()
        self.pipeline = KPipeline(model=self.model, lang_code='a', device="cuda")
            
        print("🔥 Warming up the model...")
        warmup_runs = 6
        warm_up_prompt = "Hello, we are Moe and Dal, your guides to Modal. We can help you get started with Modal, a platform that lets you run your Python code in the cloud without worrying about the infrastructure. We can walk you through setting up an account, installing the package, and running your first job."
        for _ in range(warmup_runs):
            for _ in self._stream_tts(warm_up_prompt):
                pass
        print("✅ Model warmed up!")

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
                        for chunk in self._stream_tts(prompt_msg['text'], voice=prompt_msg['voice']):
                            await audio_queue.put(chunk)
                            print(f"Sending audio data to queue: {len(chunk)} bytes")
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
            
    def _stream_tts(self, prompt: str, voice = None, speed = 1.3):

        if voice is None:
            voice = DEFAULT_VOICE

        try:
            stream_start = time.perf_counter()
            chunk_count = 0
            first_chunk_time = None

            # Generate streaming audio from the input text
            print(f"🎤 Starting streaming generation for prompt: {prompt}")
            
            for (gs, ps, chunk) in self.pipeline(
                prompt, 
                voice=voice,
                speed = speed,
            ):
                if first_chunk_time is None:
                    print(f"⏱️  Time to first chunk: {(time.perf_counter() - stream_start):.3f} seconds")
                
                print(f"gs: {gs}, ps: {ps}, chunk len: {len(chunk)}")
                chunk_count += 1
                if chunk_count % 10 == 0:  # Log every 10th chunk
                    print(f"📊 Streamed {chunk_count} chunks so far")
                
                try:
                    
                    # Ensure tensor is on CPU and convert to numpy for efficiency
                    audio_numpy = chunk.cpu().numpy()

                    audio_numpy = audio_numpy.clip(-1.0, 1.0) * 32767
                    audio_numpy = audio_numpy.astype('int16')

                    audio_segment = AudioSegment(
                        audio_numpy.tobytes(),
                        frame_rate=24000,
                        sample_width=2,
                        channels=1
                    )

                    def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
                        trim_ms = 0  # ms
                        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
                            trim_ms += chunk_size

                        return trim_ms - chunk_size # return the index of the last chunk with silence for padding

                    speech_start_idx = detect_leading_silence(audio_segment)
                    audio_segment = audio_segment[speech_start_idx:]
                    yield audio_segment.raw_data
                    
                except Exception as e:
                    print(f"❌ Error converting chunk {chunk_count}: {e}")
                    print(f"   Chunk shape: {chunk.shape if hasattr(chunk, 'shape') else 'N/A'}")
                    print(f"   Chunk type: {type(chunk)}")
                    continue  # Skip this chunk and continue
            
            final_time = time.time()
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
    kokoro_tts = modal.Cls.from_name("kokoro-tts", "KokoroTTS").with_options(scaledown_window=2)
    num_cold_starts = 50
    for _ in range(num_cold_starts):
        start_time = time.time()
        kokoro_tts().ping.remote()
        end_time = time.time()
        print(f"Time taken to ping: {end_time - start_time:.3f} seconds")
        time.sleep(10.0) # allow container to drain
    print(f"Kokoro TTS cold starts: {num_cold_starts}")
