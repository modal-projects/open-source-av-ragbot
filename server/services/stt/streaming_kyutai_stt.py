import asyncio
import base64
import time
from pathlib import Path

import modal

app = modal.App(name="streaming-kyutai-stt")

stt_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "moshi", 
        "fastapi",
        "hf_transfer", 
        "uvicorn[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_NAME = "kyutai/stt-1b-en_fr-candle"

hf_cache_vol = modal.Volume.from_name(f"{app.name}-hf-cache", create_if_missing=True)
hf_cache_vol_path = Path("/root/.cache/huggingface")
volumes = {hf_cache_vol_path: hf_cache_vol}

kyutai_server_dict = modal.Dict.from_name("kyutai-server-dict", create_if_missing=True)

MINUTES = 60

with stt_image.imports():
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from moshi.models import LMGen, loaders
    from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
    import sphn
    import tempfile

@app.cls(
    image=stt_image, 
    gpu=["L40S"], 
    volumes=volumes, 
    timeout=10 * MINUTES,
    min_containers=1,
    buffer_containers=1,
    region='us-east-1'
)
class StreamingKyutaiSTT:
    BATCH_SIZE = 1

    @modal.enter()
    def enter(self):
        import threading

        import uvicorn
        from fastapi import FastAPI

        start_time = time.monotonic_ns()

        print("Loading model...")
        snapshot_download(MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(MODEL_NAME)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        self.moshi = checkpoint_info.get_moshi(device=self.device)
        self.lm_gen = LMGen(self.moshi, temp=0, temp_text=0)

        self.mimi.streaming_forever(self.BATCH_SIZE)
        self.lm_gen.streaming_forever(self.BATCH_SIZE)

        self.text_tokenizer = checkpoint_info.get_text_tokenizer()

        self.audio_silence_prefix_seconds = checkpoint_info.stt_config.get(
            "audio_silence_prefix_seconds", 1.0
        )
        self.audio_delay_seconds = checkpoint_info.stt_config.get(
            "audio_delay_seconds", 5.0
        )
        self.padding_token_id = checkpoint_info.raw_config.get(
            "text_padding_token_id", 3
        )

        self.is_speaking = False
        self.vad_detected = False
        self.vad_signal_delay_count = 0
        self.vad_signal_delay_threshold = 3
        # warmup gpus
        for _ in range(4):
            codes = self.mimi.encode(
                torch.zeros(self.BATCH_SIZE, 1, self.frame_size).to(self.device)
            )
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
        torch.cuda.synchronize()

        print(f"Model loaded in {round((time.monotonic_ns() - start_time) / 1e9, 2)}s")

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def transcribe_websocket(ws: WebSocket):
            await ws.accept()

            # opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)
            audio_queue = asyncio.Queue()
            transcription_queue = asyncio.Queue()

            print("Session started")
            tasks = []

            # asyncio to run multiple loops concurrently within single websocket connection
            async def recv_loop(audio_queue, ws):
                """
                Receives Opus stream across websocket, appends into inbound queue.
                """
                # nonlocal opus_stream_inbound
                while True:
                    data = await ws.receive_bytes()

                    if not isinstance(data, bytes):
                        print("received non-bytes message")
                        continue
                    if len(data) == 0:
                        print("received empty message")
                        continue
                    
                    await audio_queue.put(data)

            async def inference_loop(audio_queue, transcription_queue):
                """
                Runs streaming inference on inbound data, and if any response audio is created, appends it to the outbound stream.
                """
                # nonlocal opus_stream_inbound, transcription_queue
                all_pcm_data = None
                self.is_speaking = False
                self.speaking_ended = False
                while True:
                    # await asyncio.sleep(0.001)

                    pcm = await audio_queue.get()
                    pcm = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                    async for msg in self.transcribe(pcm, all_pcm_data):
                        if isinstance(msg, str):
                            await transcription_queue.put(msg)
                        else:
                            all_pcm_data = msg

            async def send_loop(transcription_queue, ws):
                """
                Reads outbound data, and sends it across websocket
                """
                # nonlocal transcription_queue
                while True:
                    data = await transcription_queue.get()

                    # if data is None:
                    #     continue

                    # msg = b"\x01" + bytes(
                    #     data, encoding="utf8"
                    # )  # prepend "\x01" as a tag to indicate text
                    # await ws.send_bytes(msg)
                    await ws.send_text(data)
                    print(f"sending transcription text: {data}")

            # run all loops concurrently
            try:
                tasks = [
                    asyncio.create_task(recv_loop(audio_queue, ws)),
                    asyncio.create_task(inference_loop(audio_queue, transcription_queue)),
                    asyncio.create_task(send_loop(transcription_queue, ws)),
                ]
                await asyncio.gather(*tasks)

            except WebSocketDisconnect:
                print("WebSocket disconnected")
                await ws.close(code=1000)
            except Exception as e:
                print("Exception:", e)
                await ws.close(code=1011)  # internal error
                raise e
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.reset_state()

        def start_server():
            uvicorn.run(web_app, host="0.0.0.0", port=8000)

        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.tunnel_ctx = modal.forward(8000)
        self.tunnel = self.tunnel_ctx.__enter__()
        print("forwarding get / 200 at url: ", self.tunnel.url)
        websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"
        kyutai_server_dict.put("websocket_url", websocket_url)
        print(f"Websocket URL: {websocket_url}")

    def reset_state(self):
        # reset llm chat history for this input
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.is_speaking = False
        self.vad_detected = False

    async def transcribe(self, pcm, all_pcm_data):
        

        if pcm is None:
            yield all_pcm_data
            return
        if len(pcm) == 0:
            yield all_pcm_data
            return

        if pcm.shape[-1] == 0:
            yield all_pcm_data
            return

        start_time = time.monotonic_ns()
        if all_pcm_data is None:
            all_pcm_data = pcm
        else:
            all_pcm_data = np.concatenate((all_pcm_data, pcm))
        end_time = time.monotonic_ns()
        print(f"Concatenation time: {round((end_time - start_time) / 1e9, 2)}s")

        # torch_data = torch.from_numpy(all_pcm_data)
        # infer on each frame
        while all_pcm_data.shape[-1] >= self.frame_size:
            start_time = time.monotonic_ns()
            chunk = all_pcm_data[: self.frame_size]
            all_pcm_data = all_pcm_data[self.frame_size :]

            chunk = torch.from_numpy(chunk)
            with torch.no_grad():
                
                # chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
                # print(f"Chunk max: {chunk.max()}, min: {chunk.min()}")
                # chunk = torch.from_numpy(chunk)
                chunk = chunk.unsqueeze(0).unsqueeze(0)  # (1, 1, frame_size)
                chunk = chunk.expand(
                    self.BATCH_SIZE, -1, -1
                )  # (batch_size, 1, frame_size)
                chunk = chunk.to(device=self.device)
                end_time = time.monotonic_ns()
                print(f"Expanding time: {round((end_time - start_time) / 1e9, 2)}s")
                start_time = time.monotonic_ns()
                # inference on audio chunk
                codes = self.mimi.encode(chunk)

                # language model inference against encoded audio
                # for c in range(codes.shape[-1]):
                text_tokens, vad_heads = self.lm_gen.step_with_extra_heads(
                    codes
                )
                end_time = time.monotonic_ns()
                print(f"Encoding time: {round((end_time - start_time) / 1e9, 2)}s")
                
                if vad_heads:
                    pr_vad = vad_heads[2][0, 0, 0].cpu().item()
                    
                    if pr_vad > 0.5:
                        # if self.is_speaking:
                            # end of turn detected
                        # print(f"pr_vad: {pr_vad}")
                        # print("end of turn detected")
                        if not self.vad_detected:
                            self.vad_detected = True
                            self.vad_signal_delay_count = 0

                    else:
                        self.vad_detected = False

                print("text_tokens: ", text_tokens)
                    
                text_token = text_tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    start_time = time.monotonic_ns()
                    text = self.text_tokenizer.id_to_piece(text_token)
                    text = text.replace("▁", " ")
                    print("model is speaking: ", text)
                    end_time = time.monotonic_ns()
                    print(f"Text tokenization time: {round((end_time - start_time) / 1e9, 2)}s")
                    self.vad_detected = False
                    self.vad_signal_delay_count = 0
                    yield text
                elif text_token == 0:
                    self.vad_detected = False
                    self.vad_signal_delay_count = 0
                elif text_token == 3:
                    if self.vad_detected: # and self.is_speaking:
                        if self.vad_signal_delay_count == self.vad_signal_delay_threshold:
                            # self.vad_signal_delay_count = 0
                            print("end of turn detected")
                            yield "[END_OF_TURN]"
                        self.vad_signal_delay_count += 1
                    # self.is_speaking = False
        
        yield all_pcm_data

    @modal.asgi_app()
    def api(self):
        

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def transcribe_websocket(ws: WebSocket):
            await ws.accept()

            # opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)
            audio_queue = asyncio.Queue()
            transcription_queue = asyncio.Queue()

            print("Session started")
            tasks = []

            # asyncio to run multiple loops concurrently within single websocket connection
            async def recv_loop(audio_queue, ws):
                """
                Receives Opus stream across websocket, appends into inbound queue.
                """
                # nonlocal opus_stream_inbound
                while True:
                    data = await ws.receive_bytes()

                    if not isinstance(data, bytes):
                        print("received non-bytes message")
                        continue
                    if len(data) == 0:
                        print("received empty message")
                        continue
                    await audio_queue.put(data)

            async def inference_loop(audio_queue, transcription_queue):
                """
                Runs streaming inference on inbound data, and if any response audio is created, appends it to the outbound stream.
                """
                # nonlocal opus_stream_inbound, transcription_queue
                all_pcm_data = None
                self.is_speaking = False
                self.speaking_ended = False
                while True:
                    await asyncio.sleep(0.001)

                    pcm = await audio_queue.get()
                    async for msg in self.transcribe(pcm, all_pcm_data):
                        if isinstance(msg, str):
                            await transcription_queue.put(msg)
                        else:
                            all_pcm_data = msg

            async def send_loop(transcription_queue, ws):
                """
                Reads outbound data, and sends it across websocket
                """
                # nonlocal transcription_queue
                while True:
                    data = await transcription_queue.get()

                    # if data is None:
                    #     continue

                    # msg = b"\x01" + bytes(
                    #     data, encoding="utf8"
                    # )  # prepend "\x01" as a tag to indicate text
                    # await ws.send_bytes(msg)
                    await ws.send_text(data)
                    print(f"sending transcription text: {data}")

            # run all loops concurrently
            try:
                tasks = [
                    asyncio.create_task(recv_loop(audio_queue, ws)),
                    asyncio.create_task(inference_loop(audio_queue, transcription_queue)),
                    asyncio.create_task(send_loop(transcription_queue, ws)),
                ]
                await asyncio.gather(*tasks)

            except WebSocketDisconnect:
                print("WebSocket disconnected")
                await ws.close(code=1000)
            except Exception as e:
                print("Exception:", e)
                await ws.close(code=1011)  # internal error
                raise e
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.reset_state()

        return web_app

    @modal.method()
    async def transcribe_queue(self, q: modal.Queue):
        

        all_pcm_data = None

        while True:
            chunk = await q.get.aio(partition="audio")
            if chunk is None:
                await q.put.aio(None, partition="transcription")
                break

            # to avoid having to encode the audio and retrieve with OpusStreamReader:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(chunk)
                tmp.flush()
                pcm, _ = sphn.read(tmp.name)
                pcm = pcm.squeeze(0)

            async for msg in self.transcribe(pcm, all_pcm_data):
                if isinstance(msg, str):
                    await q.put.aio(msg, partition="transcription")
                else:
                    all_pcm_data = msg
