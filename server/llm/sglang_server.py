from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import sys
from typing import Final
import modal
from tqdm import tqdm
import asyncio

import requests

APP_NAME: Final[str] = "sglang-server"
MODEL_NAME: Final[str] = "Qwen/Qwen3-4B-Instruct-2507"


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

@app.cls(
    image=sglang_image,
    cpu=64,
    memory=256*1024,
    gpu=["H100!"],
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
)
class SGLangServer:
    @modal.enter(snap=True)
    def _startup(self) -> None:
        """Start the SGLang server and block until it is healthy."""

        import json
        import time
        import urllib.request
        import requests

        cmd: list[str] = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            "8092",
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

        url: str = "http://127.0.0.1:8092/v1/models"
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
                            url = "http://127.0.0.1:8092/release_memory_occupation"
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
        url = "http://127.0.0.1:8092/resume_memory_occupation"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={})
        print("Finished moving back to gpu")

        memory_usage = get_gpu_memory_usage()
        print(f"Current GPU memory usage post-resume: {memory_usage:.2f}GB")

        self.tunnel_ctx = modal.forward(8092)
        self.tunnel = await self.tunnel_ctx.__aenter__()
        print(f"SGLANG URL: {self.tunnel.url}")

    def _warmup(self) -> None:
        """Send a few warmup requests to the server."""
        import base64
        import io

        from openai import OpenAI
        from PIL import Image

        print("🚀 Warming up server...")

        client = OpenAI(base_url="http://127.0.0.1:8092/v1", api_key="123")
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
    async def register_client(self, d: modal.Dict, client_id: str):
        try:
            print(f"Registering client {client_id}")
            print(f"Tunnel URL: {self.tunnel.url}")
            d.put("url", self.tunnel.url)
            
            while not d.contains(client_id):
                await asyncio.sleep(0.100)
                
            while still_running := await d.get.aio(client_id):
                await asyncio.sleep(0.100)

        except Exception as e:
            print(f"Error registering client: {type(e)}: {e}")

    # @modal.web_server(port=8092, startup_timeout=5 * 60)
    # def serve(self) -> None:  # pragma: no cover
    #     """Web server endpoint (actual server started in _startup)."""
    #     return

    @modal.method()
    def get_response(
        self,
        prompt: str,
    ) -> str:
        import os
        import time

        import httpx
        from openai import APIConnectionError, OpenAI

        base_url: str | None = os.environ.get("MODAL_SGL_URL")
        api_key: str = "123"

        client = OpenAI(base_url=base_url, api_key=api_key)

        system_text = "You are a helpful assistant."
        messages = [
            {
                "role": "system",
                "content": system_text,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        max_attempts = 3
        base_backoff_s = 2.0
        t0 = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            try:
                result = client.chat.completions.create(
                    model=MODEL_PATH,
                    messages=messages,
                    temperature=0,
                    max_tokens=4096,
                    top_p=0.95,
                )
                break
            except (APIConnectionError, httpx.HTTPError) as exc:
                if attempt == max_attempts:
                    raise RuntimeError(
                        "OpenAI chat completion request failed after retries"
                    ) from exc
                sleep_for = max(base_backoff_s * (2 ** (attempt - 1)), 2)
                print(
                    f"retry {attempt}/{max_attempts} after {sleep_for:.1f}s due to {exc.__class__.__name__}: {exc}"
                )
                time.sleep(sleep_for)
        t1 = time.perf_counter()
        print(f"SGLang completion took {(t1 - t0)} seconds after {attempt} attempts")
        content = ""
        if hasattr(result, "choices"):
            choice = result.choices[0]
            message = getattr(choice, "message", None)
            if message is not None and hasattr(message, "content"):
                content = message.content or ""
        if isinstance(result, dict):
            content = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
        print(content)
        return content

if __name__ == "__main__":
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--single-prompt", help="Try out a single prompt", type=str)
    # group.add_argument("--prompts-file", help="File containing one prompt per line", type=str)
    # args = parser.parse_args()

    # SGLangServer = modal.Cls.from_name(APP_NAME, "SGLangServer")
    # server = SGLangServer()

    # if args.single_prompt:
    #     response = server.get_response.remote(args.single_prompt)
    #     sys.exit(0)

    # with open(args.prompts_file) as f:
    #     prompts = f.readlines()

    # # Non-modal-map approach
    # threadpoolexecutor = ThreadPoolExecutor(max_workers=10)
    # futures = []
    # for _i, prompt in enumerate(prompts):
    #     futures.append(threadpoolexecutor.submit(server.get_response.remote, prompt))
    
    # finished = 0
    # for future in tqdm(as_completed(futures), total=len(prompts), desc="Processing prompts"):
    #     finished += 1
    #     print(f"[{finished}/{len(prompts)}]")
    import time
    import uuid


    def make_request(server_cls):
        start_time = time.time()
        with modal.Dict.ephemeral() as d:
            client_id = str(uuid.uuid4())
            d.put(client_id, True)
            call_id = server_cls().register_client.spawn(d, client_id)
            while not d.contains("url"):
                time.sleep(0.100)
            sglang_url = d.get("url")
            print(f"SGLang URL: {sglang_url}")
            response = requests.post(
                f"{sglang_url}/v1/chat/completions", 
                timeout=20*60,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                }
            )
            print(response.json())
            d.put(client_id, False)
            end_time = time.time()
            print(f"Time taken to ping: {end_time - start_time:.3f} seconds")

    sglang_server = modal.Cls.from_name("sglang-server", "SGLangServer")
    num_cold_starts = 3
    for _ in range(num_cold_starts):
        make_request(sglang_server)
        time.sleep(30.0) # allow container to drain
    print(f"SGLang cold starts: {num_cold_starts}")
    # sglang_server = modal.Cls.from_name("sglang-server", "SGLangServer").with_concurrency(max_inputs=None)
    # make_request(sglang_server)
    # print("Reset max inputs to None")
