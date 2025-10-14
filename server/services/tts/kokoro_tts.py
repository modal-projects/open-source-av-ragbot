import asyncio
from pathlib import Path
import sys
# import socket
from struct import pack, unpack
import time
import re

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "kokoro>=0.9.4",
        "soundfile",
        "fastapi[standard]",
        "torchaudio",
        "transformers",
        "torch",
    )
    .add_local_dir(Path(__file__).parent / "assets", "/voice_samples")
)
app = modal.App("kokoro-tts", image=image)

with image.imports():
    import torchaudio as ta 
    from fastapi.responses import StreamingResponse
    from kokoro import KPipeline
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

_MODAL_PHONETIC_TEXT = "[Modal](/məʊdᵊl/)"
_MOE_PHONETIC_TEXT = "[Moe](/məʊ/)"
_DAL_PHONETIC_TEXT = "[Dal](/dæl/)" # alt: dæl

# kokoro_tts_dict = modal.Dict.from_name("kokoro-tts-dict", create_if_missing=True)

FORMAT = ('!I', 4)
async def sendbuffer(writer, b):
    print(f"Sending data of length {len(b)}")
    buffer = pack(FORMAT[0], len(b)) + b
    writer.write(buffer)
    await writer.drain()
    print(f"Sent data of length {len(b)}")


async def recvbuffer(reader, chunk_size: int = 1920):
    header = await reader.read(FORMAT[1])
    # print(f"Received header: {header}")
    if not header:
        yield None
    incoming_bytes = unpack(FORMAT[0], header)[0]
    if not incoming_bytes:
        yield None
    print(f"Reading data of length {incoming_bytes}")
    received_bytes = 0
    while received_bytes < incoming_bytes: # Read until we have all the data
        data = await reader.read(min(chunk_size, incoming_bytes - received_bytes))
        if not data:
            continue
        received_bytes += len(data)
        print(f"Received data of length {received_bytes}")
        yield data

@app.cls(
    gpu=["L40S", "A100", "A100-80GB"], min_containers=1, region='us-west'
)
# @modal.concurrent(max_inputs=10)
class KokoroTTS:
    @modal.enter()
    async def load(self):
        # from modal._tunnel import _forward as get_tunnel
        # kokoro_tts_dict.put("server_ready", False)
        # try:
        #     self.tunnel_manager = modal.forward(port=8000, unencrypted=True)
        #     self.tunnel = await self.tunnel_manager.__aenter__()
        # except Exception as e:
        #     print(f"❌ Error starting tunnel: {e}")
        # print(f"Tunnel started: {self.host}:{self.port}")
        try:
        
            self.model = KPipeline(lang_code='a')
            self.audio_prompt_path = "/voice_samples/kitt_voice_sample_converted_short_24000.wav"

            print("🔥 Warming up the model...")
            # warm up the model\[Kokoro](/kˈOkəɹO/)
            self._is_live = False
            warmup_runs = 20
            warm_up_prompt = "Hello, we are Moe and Dal, your guides to Modal. We can help you get started with Modal, a platform that lets you run your Python code in the cloud without worrying about the infrastructure. We can walk you through setting up an account, installing the package, and running your first job."
            for _ in range(warmup_runs):
                for _ in self._stream_tts(warm_up_prompt):
                    pass
            self._is_live = True
            print("✅ Model warmed up!")
            
            # self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.sock.bind(("0.0.0.0", self.port))
            # self.sock.listen(1)

            async def client_handler(reader, writer):
                print("Started new connection...")
                # asyncio.sleep(0.5)
                while True:
                    prompt = b""
                    async for chunk in recvbuffer(reader):
                        if chunk is None:
                            print("Received None chunk")
                            continue
                        prompt += chunk
                    
                    if not prompt:
                        await asyncio.sleep(0.001)
                        continue
                    prompt = prompt.decode("utf-8").strip()
                    

                    print(f"Received prompt: {prompt}")

                    if prompt == "<close>":
                        print("Closing connection...")
                        await sendbuffer(writer, "<close>".encode("utf-8"))
                        break
                    for chunk in self._stream_tts(prompt):
                        print(f"Sending data of length {len(chunk)}")
                        await sendbuffer(writer, chunk)
                    await sendbuffer(writer, "<tts_end>".encode("utf-8"))

                writer.close()
                await writer.wait_closed()
                print("Connection closed")

            self.server = await asyncio.start_server(client_handler, host='0.0.0.0', port=8000)
            addrs = ', '.join(str(sock.getsockname()) for sock in self.server.sockets)
            print(f'Serving on {addrs}')

        except Exception as e:
            self.exit()
            raise e
        
    @property
    def port(self):
        return self.tunnel.tcp_socket[1]
    
    @property
    def host(self):
        return self.tunnel.tcp_socket[0]
    
    @modal.method()
    async def run(self, id: str, d: modal.Dict):
        
        from modal._tunnel import _forward as get_tunnel

        
        # print(f"Address put in dict: {self.host}:{self.port}")
        server_task = None
        # d.put("server_ready", False)
        try:
            self.tunnel_manager = modal.forward(port=8000, unencrypted=True)
            self.tunnel = await self.tunnel_manager.__aenter__()
        except Exception as e:
            print(f"❌ Error starting tunnel: {e}")
        print(f"Tunnel started: {self.host}:{self.port}")
        async def run_server():
            async with self.server:
                d.put("host", self.host)
                d.put("port", self.port)
                d.put("server_ready", True)
                await self.server.serve_forever()
        try:

            server_task = asyncio.create_task(run_server())
            running = await d.get.aio(id)
            print(f"Running: {running}")
            while running:
                await asyncio.sleep(1.0)
                running = await d.get.aio(id)
                print(f"Running: {running}")
                

        except Exception as e:
            print(f"❌ Error in run: {e}")
            raise e
        finally:
            if server_task:
                server_task.cancel()
                server_task = None
            d.put("server_ready", False)

    @modal.exit()
    async def exit(self):
        await self.tunnel_manager.__aexit__(*sys.exc_info())
        print("Tunnel stopped")
    

    # @modal.fastapi_endpoint(docs=True, method="POST")
    # async def tts(self, prompt: str):
    #     # Return the audio as a streaming response with appropriate MIME type.
    #     # This allows for browsers to playback audio directly.
    #     return StreamingResponse(
    #         content=self._stream_tts(prompt),
    #         media_type="audio/wav",
    #         headers={"Content-Disposition": 'attachment; filename="output.wav"'},
    #     )

    @modal.asgi_app()
    def webapp(self):
        
        web_app = FastAPI()

        @web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):

            prompt_queue = asyncio.Queue()
            audio_queue = asyncio.Queue()

            # vad = self.VADIterator(
            #     self.silero_vad, 
            #     threshold = 0.4,
            #     sampling_rate = SAMPLE_RATE,
            #     min_silence_duration_ms = 500,
            #     speech_pad_ms = 100,
            # )

            async def recv_loop(ws, prompt_queue):
                while True:
                    data = await ws.receive_text()
                    prompt = data.strip()
                    await prompt_queue.put(prompt)
                    print(f"Received prompt: {prompt}")
                    
            async def inference_loop(prompt_queue, audio_queue):
                while True:
                    prompt = await prompt_queue.get()
                    start_time = time.perf_counter()
                    for chunk in self._stream_tts(prompt):
                        await audio_queue.put(chunk)
                    end_time = time.perf_counter()
                    print(f"Time taken to stream TTS: {end_time - start_time:.3f} seconds")

                        
            async def send_loop(audio_queue, ws):
                while True:
                    audio = await audio_queue.get()
                    await ws.send_bytes(audio)
                    print(f"sending audio data: {len(audio)} bytes")

            await ws.accept()

            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, audio_queue)),
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

        return web_app

        
    def _stream_tts(self, prompt: str):

        
        # The \b symbol in a regex pattern represents a "word boundary".
        # It matches the position between a word character (like a letter or number) and a non-word character (like a space or punctuation), 
        # or the start/end of the string. This ensures that only whole words are matched and replaced, not parts of longer words.
        prompt = re.sub(r'\bModal\b', _MODAL_PHONETIC_TEXT, prompt)
        prompt = re.sub(r'\bmodal\b', _MODAL_PHONETIC_TEXT, prompt)
        prompt = re.sub(r'\bMoe\b', _MOE_PHONETIC_TEXT, prompt)
        prompt = re.sub(r'\bmoe\b', _MOE_PHONETIC_TEXT, prompt)
        prompt = re.sub(r'\bDal\b', _DAL_PHONETIC_TEXT, prompt)
        prompt = re.sub(r'\bdal\b', _DAL_PHONETIC_TEXT, prompt)

        print(f"Phoenticized Prompt: {prompt}")

        try:

            
            stream_start = time.time()
            chunk_count = 0
            first_chunk_time = None
            # header_sent = False

            # Generate streaming audio from the input text
            print(f"🎤 Starting streaming generation for prompt: {prompt}")
            
            for (gs, ps, chunk) in self.model(
                prompt, 
                voice='am_puck',
                speed = 0.90,
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
                    # Handle tensor format - might be (batch, samples) or just (samples,)
                    if chunk.dim() > 1:
                        audio_tensor = chunk[0]  # Take first batch if batched
                    else:
                        audio_tensor = chunk
                    
                    # Ensure tensor is on CPU and convert to numpy for efficiency
                    audio_numpy = audio_tensor.cpu().numpy()
                    
                    # Convert float32 audio to int16 PCM (standard for WAV)
                    # Clamp to [-1, 1] range and scale to int16 range
                    audio_numpy = audio_numpy.clip(-1.0, 1.0)
                    pcm_data = (audio_numpy * 32767).astype('int16')
                    
                    # if not header_sent:
                    #     # Send WAV header only for the first chunk
                    #     # This is much more efficient than full WAV file per chunk
                    #     wav_header = self._create_wav_header(
                    #         sample_rate=self.model.sr,
                    #         channels=1,
                    #         bits_per_sample=16,
                    #         # Use a large data size for streaming (will be ignored by most players)
                    #         estimated_data_size=1000000  
                    #     )
                    #     yield wav_header + pcm_data.tobytes()
                    #     header_sent = True
                    # else:
                    #     # For subsequent chunks, just send raw PCM data
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
        return KokoroTTS().tts.get_web_url()
    except Exception as e:
        try:
            KokoroTTSCls = modal.Cls.from_name("kokoro-tts", "KokoroTTS")
            return KokoroTTSCls().tts.get_web_url()
        except Exception as e:
            print(f"❌ Error getting Kokoro server URL: {e}")
            return None