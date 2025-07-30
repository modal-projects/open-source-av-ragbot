# # Real-time audio transcription using Parakeet

import asyncio
import os
import sys
from pathlib import Path

import modal

app = modal.App("example-parakeet")

model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)

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
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "fastapi==0.115.12",
        "numpy<2",
        "pydub==0.25.1",
        "librosa",
        "soundfile",
    )
    .entrypoint([])  # silence chatty logs by container on start
)

END_OF_STREAM = (
    b"END_OF_STREAM_8f13d09"  # byte sequence indicating a stream is finished
)
TRANSCRIPTION_READY = (
    b"TRANSCRIPTION_READY_8f13d09"  # byte sequence indicating a stream is ready
)


@app.cls(
    volumes={"/cache": model_cache},
    gpu="a100",
    image=image,
    min_containers=1,
)
@modal.concurrent(max_inputs=14, target_inputs=10)
class Parakeet:
    @modal.enter()
    def load(self):
        import time
        from urllib.request import urlopen

        self.start_time = time.time()
        import logging

        import nemo.collections.asr as nemo_asr

        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2", map_location="cuda"
        )

        self.model.eval()
        self.end_time = time.time()
        print(
            f"🚀 Model loaded! Time taken: {self.end_time - self.start_time:.2f} seconds"
        )

        # warm up model
        start_time = time.time()
        audio_url = "https://github.com/kyutai-labs/delayed-streams-modeling/raw/refs/heads/main/audio/bria.mp3"
        # load with soundfile
        import soundfile as sf
        import librosa
        import io
        import numpy as np
        audio_data = urlopen(audio_url).read()
        audio_bytes, sample_rate = sf.read(io.BytesIO(audio_data))
        # convert to 16000 Hz
        if sample_rate != 16000:
            audio_bytes = librosa.resample(audio_bytes, orig_sr=sample_rate, target_sr=16000)
        # convert to bytes
        audio_bytes = audio_bytes.astype(np.int16).tobytes()
        
        self.transcribe.local(audio_bytes)
        end_time = time.time()
        print(f"🚀 Model warmed up! Time taken: {end_time - start_time:.2f} seconds")
        
        

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        import numpy as np


        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        with NoStdStreams():  # hide output, see https://github.com/NVIDIA/NeMo/discussions/3281#discussioncomment-2251217
            output = self.model.transcribe([audio_data])

        return output[0].text

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, WebSocket
        from fastapi import WebSocketDisconnect

        web_app = FastAPI()

        @web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            
            await ws.accept()

            audio_queue = asyncio.Queue()
            segment_queue = asyncio.Queue()
            text_queue = asyncio.Queue()

            async def receiver():
                try:
                    while True:
                        chunk = await ws.receive_bytes()
                        await audio_queue.put(chunk)
                        if chunk == END_OF_STREAM:
                            break
                except Exception as e:
                    await audio_queue.put(END_OF_STREAM)
                    raise e

            async def sender():
                try:
                    while True:
                        text = await text_queue.get()
                        if text == END_OF_STREAM:
                            await ws.send_bytes(END_OF_STREAM)
                            break
                        await ws.send_text(text)
                except Exception as e:
                    raise e

            # Use the shared preprocessor and transcriber logic
            preprocessor_task = asyncio.create_task(
                self.audio_preprocessor(audio_queue, segment_queue)
            )
            transcriber_task = asyncio.create_task(
                self.transcriber(segment_queue, text_queue)
            )
            receiver_task = asyncio.create_task(receiver())
            sender_task = asyncio.create_task(sender())

            try:
                await asyncio.gather(
                    receiver_task, preprocessor_task, transcriber_task, sender_task
                )
            except Exception as e:
                if not isinstance(e, WebSocketDisconnect):
                    print(f"Error handling websocket: {type(e)}: {e}")
                try:
                    await ws.close(code=1011, reason="Internal server error")
                except Exception as e:
                    print(f"Error closing websocket: {type(e)}: {e}")

        return web_app

    @modal.method()
    async def run_with_queue(self, q: modal.Queue):
        audio_queue = asyncio.Queue()
        segment_queue = asyncio.Queue()
        text_queue = asyncio.Queue()

        async def receiver():
            try:
                while True:
                    chunk = await q.get.aio(partition="audio")
                    await audio_queue.put(chunk)
                    if chunk == END_OF_STREAM:
                        break
            except Exception as e:
                await audio_queue.put(END_OF_STREAM)
                raise e

        async def sender():
            try:
                while True:
                    text = await text_queue.get()
                    if text == END_OF_STREAM:
                        await q.put.aio(END_OF_STREAM, partition="transcription")
                        break
                    await q.put.aio(text, partition="transcription")
            except Exception as e:
                raise e

        preprocessor_task = asyncio.create_task(
            self.audio_preprocessor(audio_queue, segment_queue)
        )
        transcriber_task = asyncio.create_task(
            self.transcriber(segment_queue, text_queue)
        )
        receiver_task = asyncio.create_task(receiver())
        sender_task = asyncio.create_task(sender())

        try:
            await q.put.aio(TRANSCRIPTION_READY, partition="transcription")
            await asyncio.gather(
                receiver_task, preprocessor_task, transcriber_task, sender_task
            )
        except Exception as e:
            print(f"Error handling queue: {type(e)}: {e}")
            return

    async def audio_preprocessor(
        self, audio_queue, segment_queue, silence_thresh=-45, min_silence_len=1000
    ):
        from pydub import AudioSegment, silence

        audio_segment = AudioSegment.empty()
        while True:
            chunk = await audio_queue.get()
            if chunk == END_OF_STREAM:
                await segment_queue.put(END_OF_STREAM)
                break
            new_audio_segment = AudioSegment(
                data=chunk,
                channels=1,
                sample_width=2,
                frame_rate=TARGET_SAMPLE_RATE,
            )
            # Only run silence detection if we have enough audio buffered
            # print(f"new_audio_segment: {len(new_audio_segment)}")
            if len(new_audio_segment) < 3 * min_silence_len:
                # print(len(new_audio_segment))
                continue

            # Only run silence detection on the new chunk
            silent_windows = silence.detect_silence(
                new_audio_segment,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
            )
            # print(silent_windows)  # Debug: can be removed
            # Adjust indices to the full buffer
            prev_len = len(audio_segment)
            audio_segment += new_audio_segment
            new_audio_segment = AudioSegment.empty()
            # print(f"new_len: {len(audio_segment)} prev_len: {prev_len}")
            if len(silent_windows) == 0:
                continue
            for i in range(len(silent_windows)):
                start, end = silent_windows[i]
                abs_start = prev_len + start
                abs_end = prev_len + end
                silent_windows[i] = (abs_start, abs_end)
            # print(silent_windows)  # Debug: can be removed
            last_window = silent_windows[-1]
            if last_window[0] < min_silence_len:
                if last_window[1] == len(audio_segment) - 1:
                    audio_segment = AudioSegment.empty()
                else:
                    audio_segment = audio_segment[
                        last_window[1] - min_silence_len / 2 :
                    ]
                continue
            segment_to_transcribe = audio_segment[
                : last_window[0] + min_silence_len / 2
            ]
            audio_segment = audio_segment[last_window[1] - min_silence_len / 2 :]
            await segment_queue.put(segment_to_transcribe.raw_data)

    async def transcriber(self, segment_queue, text_queue):
        while True:
            segment = await segment_queue.get()
            if segment == END_OF_STREAM:
                await text_queue.put(END_OF_STREAM)
                break
            try:
                text = self.transcribe(segment)
                await text_queue.put(text)
            except Exception as e:
                print("❌ Transcription error:", e)
                # Optionally, you could put an error message on the queue or skip
                continue


# ## Running transcription from a local Python client

# Next, let's test the model with a [`local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint) that streams audio data to the server and prints
# out the transcriptions to our terminal as they arrive.

# Instead of using the WebSocket endpoint like the browser frontend,
# we'll use a [`modal.Queue`](https://modal.com/docs/reference/modal.Queue)
# to pass audio data and transcriptions between our local machine and the GPU container.

AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
TARGET_SAMPLE_RATE = 16_000
CHUNK_SIZE = 16_000  # send one second of audio at a time


start_time = None
end_time = None


@app.local_entrypoint()
async def main(audio_url: str = AUDIO_URL):
    import array
    import io
    import time
    import wave
    from urllib.request import urlopen

    print(f"🌐 Downloading audio file from {audio_url}")
    audio_bytes = urlopen(audio_url).read()
    print(f"🎧 Downloaded {len(audio_bytes)} bytes")

    # Load WAV and extract samples using standard library
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_in:
        n_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        frames = wav_in.readframes(n_frames)

    # Convert frames to array based on sample width
    if sample_width == 1:
        audio_data = array.array("B", frames)  # unsigned char
    elif sample_width == 2:
        audio_data = array.array("h", frames)  # signed short
    elif sample_width == 4:
        audio_data = array.array("i", frames)  # signed int
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Downmix to mono if needed
    if n_channels > 1:
        mono_data = array.array(audio_data.typecode)
        for i in range(0, len(audio_data), n_channels):
            chunk = audio_data[i : i + n_channels]
            mono_data.append(sum(chunk) // n_channels)
        audio_data = mono_data

    # Resample to 16kHz if needed
    if frame_rate != TARGET_SAMPLE_RATE:
        ratio = TARGET_SAMPLE_RATE / frame_rate
        new_length = int(len(audio_data) * ratio)
        resampled_data = array.array(audio_data.typecode)
        for i in range(new_length):
            pos = i / ratio
            pos_int = int(pos)
            pos_frac = pos - pos_int
            if pos_int >= len(audio_data) - 1:
                sample = audio_data[-1]
            else:
                sample1 = audio_data[pos_int]
                sample2 = audio_data[pos_int + 1]
                sample = int(sample1 + (sample2 - sample1) * pos_frac)
            resampled_data.append(sample)
        audio_data = resampled_data
    audio_duration = len(audio_data) / TARGET_SAMPLE_RATE
    print(
        f"🎤 Starting Transcription for {audio_duration:.2f} seconds of streaming audio..."
    )
    start_time = time.time()
    with modal.Queue.ephemeral() as q:
        Parakeet().run_with_queue.spawn(q)
        while True:
            message = await q.get.aio(partition="transcription")
            if message:
                if message == TRANSCRIPTION_READY:
                    print("🎤 Transcription ready...")
                    break
                else:
                    raise Exception(
                        "Transcription started before READY message received."
                    )
            await asyncio.sleep(0.1)
        end_time = time.time()
        print(
            f"🚀 Transcription server started! Time taken: {end_time - start_time:.2f} seconds"
        )
        start_time = time.time()
        send = asyncio.create_task(send_audio(q, audio_data, sample_width))
        recv = asyncio.create_task(receive_text(q))
        await asyncio.gather(send, recv)
    end_time = time.time()
    print(f"✅ Transcription complete! Time taken: {end_time - start_time:.2f} seconds")


# Below are the two functions that coordinate streaming audio and receiving transcriptions.

# `send_audio` transmits chunks of audio data with a slight delay,
# as though it was being streamed from a live source, like a microphone.
# `receive_text` waits for transcribed text to arrive and prints it.


async def send_audio(q, audio_data, sample_width):
    for i in range(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[i : i + CHUNK_SIZE]
        # Convert chunk to bytes
        chunk_bytes = chunk.tobytes()
        await q.put.aio(chunk_bytes, partition="audio")
        await asyncio.sleep(0.01)
    await q.put.aio(END_OF_STREAM, partition="audio")


async def receive_text(q):
    while True:
        message = await q.get.aio(partition="transcription")
        if message == END_OF_STREAM:
            break

        print(message)


# ## Addenda

# The remainder of the code in this example is boilerplate,
# mostly for handling Parakeet's input format.


def preprocess_audio(audio_bytes: bytes) -> bytes:
    import array
    import io
    import wave

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_in:
        n_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        frames = wav_in.readframes(n_frames)

    # Convert frames to array based on sample width
    if sample_width == 1:
        audio_data = array.array("B", frames)  # unsigned char
    elif sample_width == 2:
        audio_data = array.array("h", frames)  # signed short
    elif sample_width == 4:
        audio_data = array.array("i", frames)  # signed int
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Downmix to mono if needed
    if n_channels > 1:
        mono_data = array.array(audio_data.typecode)
        for i in range(0, len(audio_data), n_channels):
            chunk = audio_data[i : i + n_channels]
            mono_data.append(sum(chunk) // n_channels)
        audio_data = mono_data

    # Resample to 16kHz if needed
    if frame_rate != TARGET_SAMPLE_RATE:
        ratio = TARGET_SAMPLE_RATE / frame_rate
        new_length = int(len(audio_data) * ratio)
        resampled_data = array.array(audio_data.typecode)

        for i in range(new_length):
            # Linear interpolation
            pos = i / ratio
            pos_int = int(pos)
            pos_frac = pos - pos_int

            if pos_int >= len(audio_data) - 1:
                sample = audio_data[-1]
            else:
                sample1 = audio_data[pos_int]
                sample2 = audio_data[pos_int + 1]
                sample = int(sample1 + (sample2 - sample1) * pos_frac)

            resampled_data.append(sample)

        audio_data = resampled_data

    return audio_data.tobytes()


def chunk_audio(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


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
