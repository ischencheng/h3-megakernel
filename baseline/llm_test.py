import json
from typing import Any

import aiohttp
import modal

app = modal.App("qwen-test")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install(
        "cuda-nsight-systems-12-9",
        "cuda-nsight-compute-12-9"
    )
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1"
        }
    )
)

@app.function(gpu="L4", image=vllm_image, timeout=600)
def chat(prompt):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=4096
    )
    outputs = llm.chat(
        [
            {"role":"user", "content":prompt}
        ],
        sampling_params=SamplingParams(max_tokens=256),
        chat_template_kwargs={"enable_thinking": False}
    )
    return outputs[0].outputs[0].text

@app.local_entrypoint()
def main():
    print(chat.remote("Write a 100 word summary about the Eagles performance this season"), flush=True)




