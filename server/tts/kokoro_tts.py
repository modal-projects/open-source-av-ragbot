import asyncio
import time
import json

import modal

from server import SERVICES_REGION

def chunk_audio(audio, desired_frame_size):
    for i in range(0, len(audio), desired_frame_size):
        yield audio[i:i + desired_frame_size]
    if len(audio) % desired_frame_size != 0:
        yield audio[-(len(audio) % desired_frame_size):]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "kokoro>=0.9.4",
        "soundfile",
        "fastapi[standard]",
        "librosa",
        "uvicorn[standard]",
    )

)
app = modal.App("kokoro-tts")

with image.imports():
    from kokoro import KPipeline
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import librosa
    import threading
    import uvicorn
    



kokoro_tts_dict = modal.Dict.from_name("kokoro-tts-dict", create_if_missing=True)

webapp = None

@app.cls(
    image=image,
    gpu=["A100", "L40S"],
    min_containers=1, 
    region=SERVICES_REGION,
    # enable_memory_snapshot=True,
    # experimental_options={"enable_gpu_snapshot": True},
)
# @modal.concurrent(max_inputs=10)
class KokoroTTS:
    @modal.enter()
    async def load(self):

        
        
        try:
        
            self.model = KPipeline(lang_code='a')
            self.audio_prompt_path = "/voice_samples/kitt_voice_sample_converted_short_24000.wav"

            print("🔥 Warming up the model...")
            # warm up the model\[Kokoro](/kˈOkəɹO/)
            warmup_runs = 20
            warm_up_prompt = "Hello, we are Moe and Dal, your guides to Modal. We can help you get started with Modal, a platform that lets you run your Python code in the cloud without worrying about the infrastructure. We can walk you through setting up an account, installing the package, and running your first job."
            for _ in range(warmup_runs):
                for _ in self._stream_tts(warm_up_prompt):
                    pass
            print("✅ Model warmed up!")


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
                                if json_data["type"] == "start_client_session":
                                    self.register_client.spawn(modal.Dict())
                                elif json_data["type"] == "prompt":
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
                    await ws.close(code=1000)
                except Exception as e:
                    print("Exception:", e)
                    await ws.close(code=1011)  # internal error
                    raise e

            def start_server():
                uvicorn.run(self.webapp, host="0.0.0.0", port=8000)

            self.server_thread = threading.Thread(target=start_server, daemon=True)
            self.server_thread.start()

            self.tunnel_ctx = modal.forward(8000)
            self.tunnel = self.tunnel_ctx.__enter__()
            print("forwarding get / 200 at url: ", self.tunnel.url)
            websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"
            kokoro_tts_dict.put("websocket_url", websocket_url)
            print(f"Websocket URL: {websocket_url}")
        

        except Exception as e:
            self.exit()
            raise e
        
    # @property
    # def port(self):
    #     return self.tunnel.tcp_socket[1]
    
    # @property
    # def host(self):
    #     return self.tunnel.tcp_socket[0]
    
    @modal.asgi_app()
    def web_endpoint(self):
        
        return self.webapp

        
    def _stream_tts(self, prompt: str, voice = None, speed = 1.3):

        if voice is None:
            voice = 'af_aoede'

        try:

            
            stream_start = time.time()
            chunk_count = 0
            first_chunk_time = None
            # header_sent = False

            # Generate streaming audio from the input text
            print(f"🎤 Starting streaming generation for prompt: {prompt}")
            
            for (gs, ps, chunk) in self.model(
                prompt, 
                voice=voice,
                speed = speed,
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    print(f"⏱️  Time to first chunk: {first_chunk_time - stream_start:.3f} seconds")
                
                print(f"gs: {gs}, ps: {ps}, chunk len: {len(chunk)}")
                chunk_count += 1
                if chunk_count % 10 == 0:  # Log every 10th chunk
                    print(f"📊 Streamed {chunk_count} chunks so far")
                
                
                # Convert torch tensor to bytes efficiently
                try:
                    # # Handle tensor format - might be (batch, samples) or just (samples,)
                    # if chunk.dim() > 1:
                    #     audio_tensor = chunk[0]  # Take first batch if batched
                    # else:
                    #     audio_tensor = chunk
                    
                    # Ensure tensor is on CPU and convert to numpy for efficiency
                    audio_numpy = chunk.cpu().numpy()

                    a, b = librosa.effects.trim(audio_numpy, top_db=20)[1]
                    a = int(a*0.9)
                    b = len(audio_numpy) #int(len(audio_numpy)-(len(audio_numpy)-b)*0.9)
                    print(f"Trimmed {len(audio_numpy)} samples to {b-a} samples")
                    audio_numpy = audio_numpy[a:b]
                    
                    # Convert float32 audio to int16 PCM (standard for WAV)
                    # Clamp to [-1, 1] range and scale to int16 range
                    audio_numpy = audio_numpy.clip(-1.0, 1.0)
                    pcm_data = (audio_numpy * 32767).astype('int16')
                    yield pcm_data.tobytes()
                    
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