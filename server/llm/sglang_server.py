from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import sys
from typing import Final
import modal
import asyncio

from server import SERVICE_REGIONS

APP_NAME: Final[str] = "sglang-server"
MODEL_NAME: Final[str] = "Qwen/Qwen3-4B-Instruct-2507"
SGLANG_PORT: Final[int] = 8092

HF_CACHE_VOL: Final[modal.Volume] = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
HF_CACHE_PATH: Final[str] = "/root/.cache/huggingface"
MODEL_PATH: Final[str] = f"{HF_CACHE_PATH}/{MODEL_NAME}"
SNAPSHOT_GPU_MEMORY_THRESHOLD: Final[float] = 10.0

sglang_image: Final[modal.Image] = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.0rc2-cu126")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "requests",
        extra_options="--no-build-isolation",
    )
    .run_commands(
        "git clone https://github.com/mattnappo/sglang.git /sglang"
    )
    .run_commands("pip uninstall -y sglang")
    .run_commands("cd /sglang && pip install -e python[all]")
    .env(
        {
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TMS_INIT_ENABLE_CPU_BACKUP": "1",
            "TORCHINDUCTOR_CACHE_DIR": f"/root/.cache/torch/",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MODAL_SGL_URL": "http://127.0.0.1:8092/v1",
            "MODAL_LOGLEVEL": "DEBUG",
        }
    )
    .workdir("/app")
)

hf_secret = None
if os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])

def get_gpu_memory_usage():
    try:
        # Run nvidia-smi with query to get memory usage in MiB
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        # Convert MiB to GiB and return as float
        return float(result.strip()) / 1024
    except subprocess.CalledProcessError:
        print("Error running nvidia-smi. Make sure NVIDIA drivers are installed.")
        return None

app: Final[modal.App] = modal.App(APP_NAME)

with sglang_image.imports():
    import json
    import time
    import urllib.request
    import requests

@app.cls(
    image=sglang_image,
    cpu=64,
    memory=256*1024,
    gpu="H100!",
    timeout=10 * 60,
    volumes={
        HF_CACHE_PATH: HF_CACHE_VOL,
    },
    # max_inputs=1,
    secrets=[hf_secret] if hf_secret is not None else [],
    enable_memory_snapshot=True,
    experimental_options={
        "enable_gpu_snapshot": True,
    },
    scaledown_window=10,
    # min_containers=1,
    region=SERVICE_REGIONS,
)
class SGLangServer:
    @modal.enter(snap=True)
    def _startup(self) -> None:
        """Start the SGLang server and block until it is healthy."""

        

        cmd: list[str] = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(SGLANG_PORT),
            "--log-level",
            "info",
            "--cuda-graph-max-bs",
            "1",
            "--context-length",
            "8192",
            # "--attention-backend",
            # "fa3",
            "--enable-memory-saver",
            "--enable-weights-cpu-backup",
        ]

        subprocess.Popen(cmd)

        url: str = f"http://127.0.0.1:{SGLANG_PORT}/v1/models"
        deadline: float = time.time() + 10 * 60
        offloaded = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url) as response:  # nosec B310
                    if response.status == 200:
                        data = json.load(response)
                        if data.get("data"):
                            print("Server is healthy 🚀 –", url)
                            self._warmup()
                            print("Offloading...")
                            url = f"http://127.0.0.1:{SGLANG_PORT}/release_memory_occupation"
                            headers = {"Content-Type": "application/json"}
                            if not offloaded:
                                response = requests.post(url, headers=headers, json={})
                            offloaded = True
                            print("Finished offloading")
                            print("Waiting until gpu memory is small")
                            t0 = time.monotonic()
                            iterations = 0
                            while True:
                                memory_usage = get_gpu_memory_usage()
                                if memory_usage is None:
                                    break
                                print(f"Current GPU memory usage: {memory_usage:.2f}GB")
                                if memory_usage < SNAPSHOT_GPU_MEMORY_THRESHOLD:
                                    print(f"GPU memory usage has dropped below threshold to {memory_usage:.2f}GB")
                                    print(f"Time taken: {time.monotonic() - t0:.2f}s")
                                    break
                                iterations += 1
                                if iterations > 10:
                                    print("GPU memory usage is still high, breaking")
                                    break
                                time.sleep(1)
                            return
            except Exception:  # pylint: disable=broad-except
                pass
            time.sleep(5)
        raise RuntimeError("Health-check failed – server did not respond in time")

    @modal.enter(snap=False)
    async def restore(self):
        print("moving back to gpu...")
        url = f"http://127.0.0.1:{SGLANG_PORT}/resume_memory_occupation"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={})
        print("Finished moving back to gpu")

        memory_usage = get_gpu_memory_usage()
        print(f"Current GPU memory usage post-resume: {memory_usage:.2f}GB")

        self.tunnel_ctx = modal.forward(SGLANG_PORT)
        self.tunnel = await self.tunnel_ctx.__aenter__()
        print(f"SGLANG URL: {self.tunnel.url}")

    def _warmup(self) -> None:
        """Send a few warmup requests to the server."""
        import base64
        import io

        from openai import OpenAI
        from PIL import Image

        print("🚀 Warming up server...")

        client = OpenAI(base_url=f"http://127.0.0.1:{SGLANG_PORT}/v1", api_key="123")
        system_text = "You are a helpful assistant."
        user_prompt = "Write me an essay about the benefits of using Modal."
        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

        # refreshed target model seems to max output tokens for warmup request
        try:
            print(f"Warmup request...")
            client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages,
                temperature=0,
                max_tokens=256,
                top_p=0.95,
            )
        except Exception as e:
            print(f"Warmup request failed: {e}")

        print("✅ Server is warmed up!")

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        try:
            print(f"Sending  url: {self.tunnel.url}")
            await d.put.aio("url", self.tunnel.url)
            
            while not await d.contains.aio("is_running"):
                await asyncio.sleep(1.0)

            print("Tunnel client is running. Waiting for it to finish.")
            
            while await d.get.aio("is_running"):
                await asyncio.sleep(1.0)

            print("Tunnel client finished.")

        except Exception as e:
            print(f"Error running tunnel client: {type(e)}: {e}")

    @modal.web_server(port=SGLANG_PORT, startup_timeout=5 * 60)
    def serve(self) -> None:  # pragma: no cover
        """Web server endpoint (actual server started in _startup)."""
        return

if __name__ == "__main__":

    import time
    import urllib.request
    import json

    def make_request(server_cls):
        start_time = time.time()
        with modal.Dict.ephemeral() as d:
            d.put("is_running", True)
            call_id = server_cls().run_tunnel_client.spawn(d)
            while not d.contains("url"):
                time.sleep(0.100)
            sglang_url = d.get("url")
            print(f"SGLang URL: {sglang_url}")

            url = f"{sglang_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = json.dumps({
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            }).encode("utf-8")

            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=20*60) as response:
                    resp_data = response.read().decode("utf-8")
                    print(json.loads(resp_data))
            except Exception as e:
                print(f"Request failed: {e}")

            d.put("is_running", False)
            time.sleep(2.0)
            call_id.cancel()
            end_time = time.time()
            print(f"Time taken to ping: {end_time - start_time:.3f} seconds")

    sglang_server = modal.Cls.from_name("sglang-server", "SGLangServer")
    num_cold_starts = 50
    for _ in range(num_cold_starts):
        make_request(sglang_server)
        time.sleep(30.0) # allow container to drain
    print(f"SGLang cold starts: {num_cold_starts}")

