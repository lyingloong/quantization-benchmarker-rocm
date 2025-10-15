from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
import torch
import torch.profiler as profiler


if __name__ == "__main__":
    torch.set_num_threads(24)  # 推理线程
    torch.set_num_interop_threads(4)  # 控制算子之间的并行

    model_path = "../QWEN3"  # 本地模型路径

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,use_fast=True)

    # 明确加载到 GPU (ROCm / CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current device: ", device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  
        device_map=None,
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    ).to(device)
    
    # config = Int8WeightOnlyConfig()
    # quantize_(model, config)
   
    inputs = tokenizer("hello", return_tensors="pt").to(device, non_blocking=True)

    # inputs = tokenizer("hello", return_tensors="pt")
    # inputs = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with profiler.record_function("inference"):
            outputs = model.generate(**inputs, max_new_tokens=100)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))