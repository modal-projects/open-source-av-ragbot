import time
import asyncio

import modal

from server import SERVICES_REGION

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
        "flashinfer-python",
        "accelerate",
        "requests",
        gpu="H100",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "1",
        "VERBOSE": "DEBUG",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "CUDA_CACHE_PATH": "/root/.cache/huggingface/.nv_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/huggingface/.inductor_cache",
        "TRITON_CACHE_DIR": "/root/.cache/huggingface.triton_cache",
        "VLLM_CACHE_ROOT": "/root/.cache/huggingface/.vllm_cache",
        })
)

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_dict = modal.Dict.from_name("vllm-dict", create_if_missing=True)

FAST_BOOT = False

app = modal.App("vllm-service")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.cls(
    image=vllm_image,
    gpu=["H200", "H100"],
    # scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    # timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers=1,
    region=SERVICES_REGION,
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
class VLLMServer():

    @modal.enter()
    async def load(self):
        import subprocess
        import threading
        import requests
        import asyncio

        def start_server():

            cmd = [
                "vllm",
                "serve",
                "--uvicorn-log-level=info",
                MODEL_NAME,
                "--served-model-name",
                MODEL_NAME,
                "llm",
                # "--host",
                # "0.0.0.0",
                "--port",
                str(VLLM_PORT),
                "--enable-chunked-prefill",
                "--max-num-seqs",
                "1",
                "--max-model-len",
                "16384",
                "--enable-prefix-caching",
                
            ]

            # enforce-eager disables both Torch compilation and CUDA graph capture
            # default is no-enforce-eager. see the --compilation-config flag for tighter control
            cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

            # assume multiple GPUs are for splitting up large matrix multiplications
            cmd += ["--tensor-parallel-size", str(N_GPU)]

            print(cmd)

            subprocess.Popen(" ".join(cmd), shell=True)

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        async def is_ready():
            try:
                response = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
                return response.status_code == 200
            except Exception as e:
                # print(f"Error checking VLLM server: {e}")
                return False

        while not await is_ready():
            await asyncio.sleep(1)

        for _ in range(4):
            start_time = time.perf_counter()
            response = requests.post(
                f"http://localhost:{VLLM_PORT}/v1/chat/completions", 
                timeout=5,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                }
            )
            print(response.json())
            end_time = time.perf_counter()
            print(f"Time taken: {end_time - start_time} seconds")


        self.tunnel_ctx = modal.forward(VLLM_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.vllm_url = self.tunnel.url
        vllm_dict.put("vllm_url", self.vllm_url)
        print(f"vLLM URL: {self.tunnel.url}")

    @modal.method()
    def ping(self):
        return "pong"

    @modal.method()
    async def register_client(self, d: modal.Dict, client_id: str):
        try:
            with modal.forward(VLLM_PORT) as tunnel:
                print(f"Registering client {client_id}")
                websocket_url = tunnel.url.replace("https://", "wss://") + "/ws"
                print(f"Websocket URL: {self.websocket_url}")
                d.put("url", websocket_url)
                
                while not d.contains("client_id"):
                    await asyncio.sleep(0.100)
                    
                while still_running := await d.get.aio("client_id"):
                    await asyncio.sleep(0.100)

        except Exception as e:
            print(f"Error registering client: {type(e)}: {e}")
    
