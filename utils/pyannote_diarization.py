import modal

app = modal.App("pyannote-test")

pyannote_volume = modal.Volume.from_name("pyannote-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("pyannote.audio")
)

with image.imports():
    import os
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

@app.function(
    image=image,
    gpu="L40S",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": pyannote_volume}
)
def diarize():
    
    # Community-1 open-source speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=os.environ["HF_TOKEN"]
    )

    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda"))

    # apply pretrained pipeline (with optional progress hook)
    with ProgressHook() as hook:
        output = pipeline(
            "/data/female_voice.wav",
            num_speakers=2,
            hook=hook
        )  

    # print the result
    print(output.speaker_diarization)
    for turn, speaker in output.speaker_diarization:
        print(f"start={turn.start:.3f}s stop={turn.end:.3f}s speaker_{speaker}")
    print(output.exclusive_speaker_diarization)
    for turn, speaker in output.exclusive_speaker_diarization:
        print(f"start={turn.start:.3f}s stop={turn.end:.3f}s speaker_{speaker}")

@app.local_entrypoint()
def main():
    diarize.remote()