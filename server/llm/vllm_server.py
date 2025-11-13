import time
import asyncio
import uuid
import modal

from server import SERVICE_REGIONS

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.10.1.1",
        "huggingface_hub[hf_transfer]==0.34",
        "flashinfer-python==0.2.14.post1",
        "accelerate",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "1",
        "VERBOSE": "DEBUG",
        # "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        # "CUDA_CACHE_PATH": "/root/.cache/.vllm/.nv_cache",
        # "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/vllm/.inductor_cache",
        # "TRITON_CACHE_DIR": "/root/.cache/vllm/.triton_cache",
        # "VLLM_CACHE_ROOT": "/root/.cache/vllm/.vllm_cache",
    })
    .uv_pip_install("requests")
    .entrypoint([])
)

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# hf_cache_vol = modal.Volume.from_name("voice-bot-hf-cache-vf2", create_if_missing=True)
# vllm_cache_vol = modal.Volume.from_name("voice-bot-vllm-cache-vf2", create_if_missing=True)
# vllm_dict = modal.Dict.from_name("vllm-dict", create_if_missing=True)

FAST_BOOT = False

app = modal.App("vllm-service")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000

with vllm_image.imports():
    import requests

@app.cls(
    image=vllm_image,
    gpu=["H100!"],
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=20 * MINUTES,  # how long should we wait for container start?
    # volumes={
    #     "/root/.cache/huggingface": hf_cache_vol,
    #     "/root/.cache/vllm": vllm_cache_vol,
    # },
    # min_containers=1,
    region=SERVICE_REGIONS,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
class VLLMServer():

    @modal.enter(snap=True)
    async def launch_vllm_server(self):
        import subprocess
        import requests

        self.tunnel_ctx = None
        self.tunnel = None
        self.vllm_url = None


        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--enable-chunked-prefill",
            "--max-num-seqs",
            "1",
            "--max-model-len",
            "16384",
            "--enable-prefix-caching",
            "--gpu-memory-utilization",
            "0.25",
            
        ]

        # enforce-eager disables both Torch compilation and CUDA graph capture
        # default is no-enforce-eager. see the --compilation-config flag for tighter control
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        # assume multiple GPUs are for splitting up large matrix multiplications
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        print(cmd)

        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )


        self.wait_for_server()

        for _ in range(4):
            start_time = time.perf_counter()
            self.ping(url_type = "local")
            end_time = time.perf_counter()
            print(f"Time taken: {end_time - start_time} seconds")

        # vllm_cache_vol.commit()
        # hf_cache_vol.commit()


    def wait_for_server(self, timeout=60 * 20):
        """Wait for the VLLM server to be ready"""
        start_time = time.time()
        count = 0
        while time.time() - start_time < timeout:
            try:
                if self.healthcheck():
                    print("VLLM server is ready!")
                    return
                count += 1
            except Exception:
                pass
            if count % 10 == 0:
                print("VLLM server NOT ready!")
            time.sleep(1)
        raise TimeoutError("VLLM server failed to start within timeout period")

    def healthcheck(self):
        """
        Perform a healthcheck on the VLLM server
        Returns True if healthy, False otherwise
        """
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # @modal.enter(snap=False)
    # async def setup_tunnel(self):
        
    #     # vllm_cache_vol.reload()
    #     # hf_cache_vol.reload()

    #     self.tunnel_ctx = modal.forward(VLLM_PORT)
    #     self.tunnel = await self.tunnel_ctx.__aenter__()
    #     self.vllm_url = self.tunnel.url
    #     print(f"vLLM URL: {self.vllm_url}")

    #     for _ in range(4):
    #         start_time = time.perf_counter()
    #         self.ping()
    #         end_time = time.perf_counter()
    #         print(f"Time taken: {end_time - start_time} seconds")

    # @modal.method()
    def ping(self, url_type: str = "local"):
        if url_type == "local":
            base_url = f'http://localhost:{VLLM_PORT}'
        elif url_type == "tunnel":
            if self.vllm_url is None:
                raise Exception("VLLM URL is not set")
            base_url = self.vllm_url
        else:
            raise ValueError(f"Invalid URL type: {url_type}")
        
        response = requests.post(
            f"{base_url}/v1/chat/completions", 
            timeout=20*60,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            }
        )
        print(response.json())
        return response.json()

    @modal.exit()
    async def exit(self):
        if self.tunnel_ctx:
            await self.tunnel_ctx.__aexit__(None, None, None)
            self.tunnel_ctx = None
            self.tunnel = None
            self.vllm_url = None

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        try:
            print(f"Sending websocket url: {self.websocket_url}")
            await d.put.aio("url", self.websocket_url)
            
            while True:
                asyncio.sleep(1.0)

        except Exception as e:
            print(f"Error running tunnel client: {type(e)}: {e}")
    
# warm up snapshots if needed
if __name__ == "__main__":
    vllm_server = modal.Cls.from_name("vllm-service", "VLLMServer")
    num_cold_starts = 5
    for _ in range(num_cold_starts):
        start_time = time.time()
        with modal.Dict.ephemeral() as d:
            client_id = str(uuid.uuid4())
            d.put(client_id, True)
            call_id = vllm_server().run_tunnel_client.spawn(d, client_id)
            while not d.contains("url"):
                time.sleep(0.100)
            vllm_url = d.get("url")
            print(f"vLLM URL: {vllm_url}")
            response = requests.post(
                f"{vllm_url}/v1/chat/completions", 
                timeout=20*60,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                }
            )
            print(response.json())
            call_id.cancel()
            end_time = time.time()
            print(f"Time taken to ping: {end_time - start_time:.3f} seconds")
            time.sleep(10.0) # allow container to drain
    print(f"vLLM cold starts: {num_cold_starts}")
