import asyncio
import os
import sys
import time
import json
import base64

import modal

from server import SERVICES_REGION

app = modal.App("parakeet-transcription")

model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)
parakeet_dict = modal.Dict.from_name("parakeet-dict", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",  # cache directory for Hugging Face models
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
            "TORCH_HOME": "/cache",
        }
    )
    .apt_install("ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "fastapi==0.115.12",
        "numpy<2",
        "torchaudio",
        "soundfile",
        "uvicorn[standard]",
    )
    .entrypoint([])  # silence chatty logs by container on start
)

SAMPLE_RATE = 16000
MIN_AUDIO_SEGMENT_DURATION_SAMPLES = int(SAMPLE_RATE / 2)
VAD_CHUNK_SIZE = 512
UVICORN_PORT = 8000

def chunk_audio(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]

def int2float(audio_data):
    abs_max = np.abs(audio_data).max()
    audio_data = audio_data.astype('float32')
    if abs_max > 0:
        audio_data *= 1/32768
    audio_data = audio_data.squeeze()  # depends on the use case
    return audio_data

def _bytes_to_torch(data):
    data = np.frombuffer(data, dtype=np.int16)
    data = torch.from_numpy(int2float(data))
    return data

with image.imports():
    import numpy as np
    import logging
    import nemo.collections.asr as nemo_asr
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
    from urllib.request import urlopen
    import torch
    import threading
    import uvicorn
    from fastapi import FastAPI


@app.cls(
    volumes={"/cache": model_cache}, 
    gpu=["A100"], 
    image=image,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region=SERVICES_REGION,
    # uncomment min containers for testing
    # min_containers=1,
    scaledown_window=10,
)
@modal.concurrent(max_inputs=20)
class Transcriber:

    @modal.enter(snap=True)
    def load(self):
        
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None
        
        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.use_vad = False
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )


        self.model.to(torch.bfloat16)
        self.model.eval()
        # Configure decoding strategy
        if self.model.cfg.decoding.strategy != "beam":
            self.model.cfg.decoding.strategy = "greedy_batch"
            self.model.change_decoding_strategy(self.model.cfg.decoding)

        # torchaudio.set_audio_backend("soundfile")

        self.silero_vad, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks
        ) = utils

        # warm up gpu
        AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
        audio_bytes = urlopen(AUDIO_URL).read()
        if audio_bytes.startswith(b"RIFF"):
            audio_bytes = audio_bytes[44:]
        
        # Convert raw bytes to int16 numpy array first
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Then chunk the audio data (not the raw bytes)
        chunk_size_seconds = 1
        chunk_size = SAMPLE_RATE * chunk_size_seconds  # 5 seconds at 16kHz
        
        audio_chunks = [torch.from_numpy(audio_data[i:i+chunk_size]) for i in range(0, len(audio_data), chunk_size)]
        audio_chunks = audio_chunks[:10]

        times = []
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
            for chunk in audio_chunks:
                start_time = time.perf_counter()
                self.model.transcribe(chunk)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        print(f"Warmup transcription quantile values ({chunk_size_seconds} second chunks):")
        print(f"p5: {np.percentile(times, 5)}")
        print(f"p50: {np.percentile(times, 50)}")
        print(f"p95: {np.percentile(times, 95)}")

        print("GPU warmed up")

    @modal.enter(snap=False)
    def _start_server(self):

        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):

            audio_queue = asyncio.Queue()

            transcription_queue = asyncio.Queue()
            vad = None
            if self.use_vad:
                vad = self.VADIterator(
                    self.silero_vad, 
                    threshold = 0.4,
                    sampling_rate = 16000,
                    min_silence_duration_ms = 250,
                    speech_pad_ms = 100,
                )
                

            async def recv_loop(ws, audio_queue):
                audio_buffer = bytearray()
                while True:
                    msg = await ws.receive_text()
                    try:
                        json_data = json.loads(msg)
                        if "type" in json_data:
                            if json_data["type"] == "start_client_session":
                                self.register_client.spawn(modal.Dict())
                            if json_data["type"] == "set_vad":
                                self.use_vad = json_data["vad"]
                                continue
                            elif json_data["type"] == "audio":
                                data = json_data["audio"]
                                data = base64.b64decode(data.encode('utf-8'))
                            else:
                                continue
                        else:
                            continue
                    except Exception as e:
                        continue

                    if self.use_vad:
                        audio_buffer.append(data)
                        if len(audio_buffer) > VAD_CHUNK_SIZE:
                            await audio_queue.put(audio_buffer[:VAD_CHUNK_SIZE])
                            audio_buffer = audio_buffer[VAD_CHUNK_SIZE:]

                    await audio_queue.put(data)
                    
            async def inference_loop(audio_queue, transcription_queue, vad=None):
                all_audio_data = None
                start_idx = None
                end_idx = None
                while True:
                    
                    audio_data = await audio_queue.get()
                    audio_data = _bytes_to_torch(audio_data)

                    if not vad:
                        start_time = time.perf_counter()
                        
                        # audio_segment = all_audio_data[start_idx:end_idx]
                        transcript = self.transcribe(audio_data)
                        await transcription_queue.put(transcript)

                        end_time = time.perf_counter()
                        print(f"time taken to transcribe audio segment: {end_time - start_time} seconds")
                    else:


                        # collect in torch array
                        if all_audio_data is None:
                            all_audio_data = audio_data
                        else:
                            all_audio_data = torch.cat([all_audio_data, audio_data])
                        
                        start_time = time.perf_counter()
                        speech_time_stamps =vad(
                            audio_data, # only need to pass in new data
                        )
                        end_time = time.perf_counter()
                        print(f"time taken to get speech timestamps: {end_time - start_time} seconds")
                        
                        # no speech detected
                        if not speech_time_stamps:
                            continue
                        
                        # start of speech detected
                        if speech_time_stamps.get("start") and start_idx is None:
                            start_idx = speech_time_stamps["start"]
                            print(f"speech started at {start_idx}")
                        
                        # end of speech detected
                        if speech_time_stamps.get("end"):
                            end_idx = speech_time_stamps["end"]
                            vad.reset_states()

                            # failback if start of speech not set
                            if start_idx is None:
                                start_idx = 0
                            
                            # don't transcribe if speech is too short
                            if end_idx - start_idx < MIN_AUDIO_SEGMENT_DURATION_SAMPLES:
                                end_idx = None # don't reset start_idx
                                continue
                            
                            start_time = time.perf_counter()
                            
                            audio_segment = all_audio_data[start_idx:end_idx]
                            transcript = self.transcribe(audio_segment)
                            await transcription_queue.put(transcript)

                            end_time = time.perf_counter()
                            print(f"time taken to transcribe audio segment: {end_time - start_time} seconds")

                            # feed leftover audio through vad and capture and speech detection
                            # take largest multiple of VAD_CHUNK_SIZE
                            samples_remaining = (len(all_audio_data) - end_idx) // VAD_CHUNK_SIZE * VAD_CHUNK_SIZE
                            all_audio_data = all_audio_data[-samples_remaining:]

                            start_idx = None
                            end_idx = None
                            
                            # this loop only captures the first start time which will be the
                            # start of the next segment
                            for chunk in chunk_audio(all_audio_data, VAD_CHUNK_SIZE):
                                speech_time_stamps = vad(chunk)
                                if not speech_time_stamps:
                                    continue
                                if speech_time_stamps.get("start") and start_idx is None:
                                    start_idx = speech_time_stamps["start"]
                                    print(f"speech started at {start_idx}")
                                if speech_time_stamps.get("end"):
                                    print(f"full speech found in remainging audio")
                                    vad.reset_states()

                        
            async def send_loop(transcription_queue, ws):
                while True:
                    transcript = await transcription_queue.get()
                    print(f"sending transcription data: {transcript}")
                    await ws.send_text(transcript)

            await ws.accept()

            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, audio_queue)),
                    asyncio.create_task(inference_loop(audio_queue, transcription_queue, vad)),
                    asyncio.create_task(send_loop(transcription_queue, ws)),
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
            uvicorn.run(self.web_app, host="0.0.0.0", port=UVICORN_PORT)

        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.tunnel_ctx = modal.forward(UVICORN_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"
        print(f"Websocket URL: {self.websocket_url}")

    @modal.method()
    async def register_client(self, d: modal.Dict, client_id: str):
        try:
            print(f"Registering client {client_id} for websocket url: {self.websocket_url}")
            d.put("url", self.websocket_url)
            
            while not d.contains(client_id):
                await asyncio.sleep(0.100)
                
            while still_running := await d.get.aio(client_id):
                await asyncio.sleep(0.100)

        except Exception as e:
            print(f"Error registering client: {type(e)}: {e}")

    def transcribe(self, audio_data) -> str:

        with NoStdStreams():  # hide output, see https://github.com/NVIDIA/NeMo/discussions/3281#discussioncomment-2251217
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
                output = self.model.transcribe([audio_data])

        return output[0].text

    @modal.asgi_app()
    def webapp(self):
        return self.web_app

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


class NoStdStreams(object):
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush(), self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


# warm up snapshots if needed
if __name__ == "__main__":
    parakeet_stt = modal.Cls.from_name("parakeet-transcription", "Transcriber")
    num_cold_starts = 5
    for _ in range(num_cold_starts):
        start_time = time.time()
        parakeet_stt().ping.remote()
        end_time = time.time()
        print(f"Time taken to ping: {end_time - start_time:.3f} seconds")
        time.sleep(30.0) # allow container to drain
    print(f"Parakeet STT cold starts: {num_cold_starts}")
