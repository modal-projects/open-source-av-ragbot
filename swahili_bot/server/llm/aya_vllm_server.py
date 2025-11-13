"""
Aya-101 Swahili LLM Service

This service uses CohereForAI's Aya-101 model with vLLM for optimized inference.
Aya-101 is a multilingual model with strong Swahili language support.
"""

import modal
import subprocess
import time
import requests
from pathlib import Path

# Import regional configuration
from swahili_bot.server.common.const import SERVICE_REGIONS

# Modal app configuration
app = modal.App("swahili-aya-llm")

# Model configuration
MODEL_NAME = "CohereForAI/aya-101"
VLLM_PORT = 8000

# GPU image with vLLM
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.6.post1",
        "requests==2.32.3",
    )
)


def wait_for_server(port: int, timeout: int = 300):
    """Wait for vLLM server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                print(f"✓ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not start within {timeout} seconds")


@app.cls(
    image=vllm_image,
    gpu="H100",  # H100 for optimal performance with this 13B model
    timeout=30 * 60,  # 30 minutes
    container_idle_timeout=5 * 60,  # 5 minutes
    enable_memory_snapshot=True,
    allow_concurrent_inputs=10,
    regions=SERVICE_REGIONS,
)
class AyaLLM:
    @modal.enter(snap=True)
    def start_vllm_server(self):
        """Start the vLLM server with Aya-101 model."""
        print(f"Starting vLLM server with {MODEL_NAME}...")

        # vLLM configuration optimized for Aya-101
        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            MODEL_NAME,
            "--served-model-name",
            "aya-101",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            "2048",  # Aya-101 supports longer contexts, but limit for speed
            "--gpu-memory-utilization",
            "0.90",  # Use most of GPU memory
            "--dtype",
            "auto",  # Auto-detect best dtype (will use FP16/BF16)
            "--max-num-seqs",
            "4",  # Allow some batching for efficiency
            "--enable-chunked-prefill",  # Faster prefill
            "--enable-prefix-caching",  # Cache system prompts
        ]

        # Start vLLM server
        self.vllm_process = subprocess.Popen(cmd)

        # Wait for server to be ready
        wait_for_server(VLLM_PORT)

        # Warmup the model
        print("Warming up model with sample requests...")
        self._warmup()

    def _warmup(self, num_requests: int = 3):
        """Warmup the model with sample requests."""
        import openai

        client = openai.OpenAI(
            api_key="dummy",  # vLLM doesn't need real API key
            base_url=f"http://localhost:{VLLM_PORT}/v1",
        )

        for i in range(num_requests):
            try:
                start = time.time()
                response = client.chat.completions.create(
                    model="aya-101",
                    messages=[
                        {"role": "system", "content": "Wewe ni msaidizi wa kirafiki."},
                        {"role": "user", "content": "Habari yako?"},
                    ],
                    max_tokens=50,
                    temperature=0.7,
                )
                latency = time.time() - start
                print(f"Warmup {i+1}/{num_requests}: {latency:.2f}s - {response.choices[0].message.content[:50]}")
            except Exception as e:
                print(f"Warmup request {i+1} failed: {e}")

    @modal.enter(snap=False)
    async def create_tunnel(self):
        """Create Modal Tunnel for direct access."""
        # Create tunnel
        self.tunnel_ctx = modal.forward(VLLM_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.base_url = self.tunnel.url + "/v1"

        print(f"vLLM API available at: {self.base_url}")

    @modal.method()
    async def run_tunnel_client(self, url_dict: modal.Dict):
        """Method to expose the API URL via Modal Dict."""
        await url_dict.put.aio("url", self.base_url)
        print(f"Published API URL: {self.base_url}")

        # Keep alive until cancelled
        import asyncio

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("Tunnel client cancelled")

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 150):
        """Simple generation method for testing."""
        import openai

        client = openai.OpenAI(
            api_key="dummy",
            base_url=f"http://localhost:{VLLM_PORT}/v1",
        )

        response = client.chat.completions.create(
            model="aya-101",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        return response.choices[0].message.content


@app.local_entrypoint()
def main():
    """Test the Aya-101 service."""
    llm = AyaLLM()

    # Test generation
    test_prompt = "Eleza kwa kifupi historia ya Tanzania."
    print(f"\nTest prompt: {test_prompt}")
    response = llm.generate.remote(test_prompt)
    print(f"Response: {response}")

    print("\nService is running. API URL:", llm.base_url)
