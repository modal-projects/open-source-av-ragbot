from pathlib import Path
import io

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "chatterbox-tts @ git+https://github.com/fakerybakery/better-chatterbox@fix-cuda-issue",
        "fastapi[standard]",
        "torchaudio",
        "transformers",
        "torch",
    )
    .add_local_dir(Path(__file__).parent / "assets", "/voice_samples")
)
app = modal.App("chatterbox-tts", image=image)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from fastapi.responses import Response

@app.cls(
    gpu="h100", scaledown_window=60 * 5, enable_memory_snapshot=True, min_containers=1
)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.audio_prompt_path = "/voice_samples/erik_voice_sample_24000.wav"

    @modal.fastapi_endpoint(docs=True, method="POST")
    def tts(self, prompt: str):
        # Generate audio waveform from the input text
        print(f"Generating audio for prompt: {prompt}")
        wav = self.model.generate(
            prompt,
            audio_prompt_path=self.audio_prompt_path,
        )

        # Create an in-memory buffer to store the WAV file
        buffer = io.BytesIO()

        # Save the generated audio to the buffer in WAV format
        # Uses the model's sample rate and WAV format
        ta.save(
            buffer,
            wav,
            self.model.sr,
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16,
        )

        print("Chatterbox response generated.")
        buffer.seek(0)
        # def generate_audio_chunks(buffer):
        #     buffer.seek(0)  # Reset to beginning
        #     while True:
        #         chunk = buffer.read(8192)  # Read in chunks
        #         if not chunk:
        #             break
        #         yield chunk

        # Return the audio as a streaming response with appropriate MIME type.
        # This allows for browsers to playback audio directly.
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="output.wav"'},
        )
