import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
        "flashinfer-python",
        # "torch",
        "accelerate",
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
        })  # faster model transfers
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


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers=1,
    region='us-east-1',
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess
    import threading

    def start_server():

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
            # "--max-num-seqs",
            # "1",
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

    tunnel_ctx = modal.forward(VLLM_PORT)
    tunnel = tunnel_ctx.__enter__()
    print("forwarding get / 200 at url: ", tunnel.url)
    vllm_dict.put("vllm_url", tunnel.url)
    print(f"vLLM URL: {tunnel.url}")
    

