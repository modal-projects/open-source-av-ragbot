import os
from pathlib import Path

import modal


CACHE_VOLUME = modal.Volume.from_name("realtimestt-cache", create_if_missing=True)
CACHE_PATH = "/cache"
cache = {CACHE_PATH: CACHE_VOLUME}

py_version = "3.12"
ld_libray_path = f"/usr/local/lib/python{py_version}/site-packages/nvidia/cudnn/lib"

MINUTES = 60

image = (
    modal.Image.debian_slim(python_version=py_version)  # matching ld path
    # update locale as required by onnx
    .env(
        {
            "LD_LIBRARY_PATH": ld_libray_path,
            "TORCH_HOME": CACHE_PATH,
            "HF_HUB_DISABLE_PROGRESS_BARS": "True" # helps with tqdm._lock thread safety issues
        }
    )
    # install system dependencies
    .apt_install("ffmpeg","python3-dev","portaudio19-dev","git")
    # install Python dependencies
    .uv_pip_install(
        "fastapi",
        "huggingface_hub[hf_xet]",
        "onnxruntime-gpu",
        "torch",
        "shortuuid",
        "realtimestt",
        "aiologic",
    )
)


app = modal.App("realtime-stt")

# default text_detected function, will be replaced with something that sends the message out to our system
def text_detected(text):
    text = f"\r{text}"
    print(f"\r{text}", flush=True, end='')

_recorder_config = {
    'download_root': CACHE_PATH,
    'spinner': False,
    'use_microphone': False,
    'model': 'large-v3',
    'language': 'en',
    'silero_sensitivity': 0.05,
    'silero_use_onnx': True,
    'silero_deactivity_detection': True,
    "webrtc_sensitivity": 3,
    'post_speech_silence_duration': 0.45,
    'early_transcription_on_silence': 0.2,
    'min_length_of_recording': 0,
    'min_gap_between_recordings': 0,
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0,
    'realtime_model_type': 'tiny.en',
    'on_realtime_transcription_stabilized': text_detected,      
    'initial_prompt': "Incomplete thoughts should end with '...'. Examples of complete thoughts: 'The sky is blue.' 'She walked home.' Examples of incomplete thoughts: 'When the sky...' 'Because he...'"

}


with image.imports():
    import asyncio
    import threading
    import aiologic
    from aiologic import QueueEmpty
    import RealtimeSTT
    import time
    import json
    from huggingface_hub import snapshot_download


def feed_audio_thread(recorder, queue, stop_event):
    """Thread function to read audio data from queue and feed it to the recorder."""

    try:
        print(f"Running audio feed thread")
        while not stop_event.is_set():
            # Read a chunk of audio data from the queue
            data = None
            try:
                if queue:
                    data = queue.green_get(timeout=1.0)
                else:
                    time.sleep(0.010)
            except QueueEmpty:
                continue
            except Exception as e:
                print(f"Error in feed_audio_thread: {e}")
                continue
            # Feed the audio data to the recorder
            if data:
                recorder.feed_audio(data)   
    except Exception as e:
        print(f"\nfeed_audio_thread encountered an error: {e}")
    finally:
        print("\nFinished feeding audio.")

def recorder_transcription_thread(recorder, queue, stop_event):
    """Thread function to handle transcription and process the text."""

    transcribed_text = []
    
    def process_text(full_sentence):
        """Callback function to process the transcribed text."""
        if full_sentence and full_sentence.strip():  # Only add non-empty sentences
            transcribed_text.append(full_sentence)
            print("\nTranscribed text:", full_sentence)
            try:
                msg = {
                    "type": "final_transcript",
                    "text": full_sentence
                }
                queue.green_put(json.dumps(msg)) 
            except Exception as e:
                print(f"Error in process_text while sending to sync queue: {type(e)}: {e}")
                pass

    def text_detected(text):

        if text:
            try:
                msg = {
                    "type": "realtime_transcript",
                    "text": text
                }
                queue.green_put(json.dumps(msg))

            except Exception as e:
                print(f"Error in text_detected while sending to sync queue: {type(e)}: {e}")

    recorder.on_realtime_transcription_stabilized = text_detected
    
    try:
        print("Starting transcription thread")
        # Process transcriptions until the file is done and no more text is coming
        while not stop_event.is_set():
            # Get transcribed text and process it using the callback
            recorder.text(process_text)
            time.sleep(0.01)  # Small sleep to prevent CPU overuse
        
    except Exception as e:
        print(f"\ntranscription_thread encountered an error: {e}")
    finally:
        print("\nTranscription thread exiting.")

END_OF_STREAM = (
    b"END_OF_STREAM_8f13d09"  # byte sequence indicating a stream is finished
)


@app.cls(
    image=image,
    gpu=["A100-80GB"],
    volumes=cache,
    min_containers=1,
    timeout= 60 * MINUTES,
)
@modal.concurrent(max_inputs=1)
class Transcriber:

    @modal.enter()
    async def enter(self):

        # download models and pass paths to threads to avoid hf_hub + tqdm thread safety issues when
        # we load the models via hf_hub inside the threads
        if not os.path.exists(CACHE_PATH + f"/models--Systran--faster-whisper-{_recorder_config['model']}"):
        
            _recorder_config['model'] = snapshot_download(
                f"Systran/faster-whisper-{_recorder_config['model']}", 
                local_files_only=False, 
                cache_dir=CACHE_PATH,
            )
       
        if not os.path.exists(CACHE_PATH + f"/models--Systran--faster-whisper-{_recorder_config['realtime_model_type']}"):
            _recorder_config['realtime_model_type'] = snapshot_download(
            f"Systran/faster-whisper-{_recorder_config['realtime_model_type']}", 
            local_files_only=False, 
            cache_dir=CACHE_PATH,
        )

        self.stt_models = {}
        self.audio_queues = {}
        self.transcription_queues = {}
        self.stop_events = {}

    async def _initialize(self):
        stt_model = RealtimeSTT.AudioToTextRecorder(**_recorder_config)
        print("model loaded")

        audio_queue = aiologic.SimpleQueue()
        transcription_queue = aiologic.SimpleQueue()

        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=feed_audio_thread, 
            args=(
                stt_model, 
                audio_queue, 
                stop_event
            )
        )
        audio_thread.start()

        # Start the transcription_thread
        transcription_thread = threading.Thread(
            target=recorder_transcription_thread,
            args=(
                stt_model, 
                transcription_queue, 
                stop_event,
            )
        )
        transcription_thread.start()

        print(f"RealtimeSTT initialized....")

        return stt_model, audio_queue, transcription_queue, stop_event

    @modal.asgi_app()
    def webapp(self):
        from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        @web_app.websocket("/ws")
        async def transcribe_websocket(ws: WebSocket):
            stt_model, audio_queue, transcription_queue, stop_event = await self._initialize()
            await ws.accept()
            print("WebSocket accepted")
            async def _send_transcription_to_client(queue, ws):
                print("Send transcription task started (in function)")
                while True:
                    try:
                        text = await queue.async_get()
                    except QueueEmpty:
                        continue
                    except Exception as e:
                        print(f"Error in _send_transcription_to_client while getting from queue: {type(e)}: {e}")
                        raise e
                        
                    try:
                        await ws.send_text(text)

                    except Exception as e:
                        print(f"Error in _send_transcription_to_client while sending to websocket: {type(e)}: {e}")
                        raise e

            async def _recv_audio_from_client(queue, ws):
                print("Recv audio task started (in function)")
                while True:
                    try:
                        data = await ws.receive_bytes()
                        await queue.async_put(data)
                    except Exception as e:
                        print(f"Error in _recv_audio_from_client while receiving from websocket: {type(e)}: {e}")
                        raise e

            _recv_audio_task = asyncio.create_task(_recv_audio_from_client(audio_queue, ws))
            _send_transcription_task = asyncio.create_task(_send_transcription_to_client(transcription_queue, ws))

            print("Send transcription task started")

            try:
                await asyncio.gather(_recv_audio_task, _send_transcription_task)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                await ws.close(code=1000)
            except Exception as e:
                print("Exception:", e)
                await ws.close(code=1011)  # internal error
                raise e
            finally:
                stop_event.set()
                print("Stop event set")
                if _send_transcription_task is not None:
                    _send_transcription_task.cancel()
                if _recv_audio_task is not None:
                    _recv_audio_task.cancel()
                stt_model.stop()
                stt_model.shutdown()
                print("Send transcription task cancelled")
                await asyncio.sleep(3.0)

                print("Send transcription task finished")

        return web_app
            
        
# web_image = (
#     modal.Image.debian_slim(python_version="3.12")
#     .pip_install("fastapi")
#     .add_local_dir(
#         Path(__file__).parent.parent / "frontend" /"streaming-realtime-stt-frontend", "/root/frontend"
#     )
# )

# @app.cls(image=web_image)
# @modal.concurrent(max_inputs=1000)
# class WebServer:
#     @modal.asgi_app()
#     def web(self):
#         from fastapi import FastAPI, Response, WebSocket
#         from fastapi.responses import HTMLResponse
#         from fastapi.staticfiles import StaticFiles

#         web_app = FastAPI()
#         web_app.mount("/static", StaticFiles(directory="frontend"))

#         @web_app.get("/status")
#         async def status():
#             return Response(status_code=200)

#         # serve frontend
#         @web_app.get("/")
#         async def index():
#             return HTMLResponse(content=open("frontend/index.html").read())

#         return web_app