from benchmark_config import BenchmarkConfig
from model_benchmarker import ModelBenchmarker
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path: str, device: str, quantize: bool = False):
    """加载模型和Tokenizer的示例函数"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  
        device_map=None,
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    ).to(device)
    
    # 可选的量化步骤
    if quantize:
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            quantize_(model, Int8WeightOnlyConfig())
            print("已应用INT8量化")
        except ImportError:
            print("警告: 未安装torchao，无法进行量化")
    
    return model, tokenizer

def main():
    # 1. 配置测试参数
    config = BenchmarkConfig(
        num_runs=5,
        warmup_runs=2,
        max_new_tokens_list=[50, 100],
        device="cuda",
        num_threads=24,
        num_interop_threads=4
    )
    
    # 2. 加载模型 (可替换为任何模型)
    model_path = "../QWEN3"  # 替换为你的模型路径
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # 加载FP16模型
    model_fp16, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        device=device,
        quantize=False
    )
    
    # 3. 运行FP16模型测试
    benchmarker_fp16 = ModelBenchmarker(
        model=model_fp16,
        tokenizer=tokenizer,
        config=config
    )
    results_fp16 = benchmarker_fp16.run_all_tests(model_config_name="FP16")
    
    # 打印FP16模型结果
    results_fp16.print_summary()
    # 可选择打印详细结果
    # results_fp16.print_detailed()
    
    # 运行性能分析 (可选)
    benchmarker_fp16.run_profiler(
        input_text="Hello world!", 
        max_new_tokens=100,
        output_file="fp16_profile_trace.json"
    )
    
    # 4. 测试量化模型 (可选)
    model_int8, _ = load_model_and_tokenizer(
        model_path=model_path,
        device=device,
        quantize=True
    )
    
    benchmarker_int8 = ModelBenchmarker(
        model=model_int8,
        tokenizer=tokenizer,
        config=config
    )
    results_int8 = benchmarker_int8.run_all_tests(model_config_name="INT8")
    
    # 打印INT8模型结果
    results_int8.print_summary()

if __name__ == "__main__":
    main()
    