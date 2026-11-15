import sys

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=4096
)

outputs = llm.chat(
    [{"role": "user", "content": sys.argv[1]}],
    sampling_params=SamplingParams(max_tokens=256)
)