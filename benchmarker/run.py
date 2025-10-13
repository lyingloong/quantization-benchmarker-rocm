from benchmark_config import BenchmarkConfig
from model_benchmarker import ModelBenchmarker
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def load_model_and_tokenizer(model_path: str, device: str, quantize: bool = False):
    print("[load_model_and_tokenizer] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  
        device_map=None,
        attn_implementation="sdpa",
        dtype=torch.float16
    ).to(device)
    
    if quantize:
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            quantize_(model, Int8WeightOnlyConfig())
            print("[load_model_and_tokenizer] Model quantized to int8.")
        except ImportError:
            print("[load_model_and_tokenizer] torchao not installed, skipping quantization.")
    
    return model, tokenizer

def main():
    # 设置随机种子
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="run benchmark")
    parser.add_argument("--model_name", type=str, default="Model", help="name of the model")
    parser.add_argument("--model_path", type=str, required=True, help="path of the model")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the model")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize the model")
    parser.add_argument("--num_runs", type=int, default=5, help="number of benchmark runs")
    parser.add_argument("--warmup_runs", type=int, default=2, help="number of warmup runs")
    # parser.add_argument("--max_new_tokens_list", type=str, default="50,100", help="comma-separated list of max_new_tokens")
    parser.add_argument("--num_threads", type=int, default=24, help="number of CPU threads")
    parser.add_argument("--num_interop_threads", type=int, default=4, help="number of interop threads")
    args = parser.parse_args()

    config = BenchmarkConfig(
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        max_new_tokens_list=[50, 100],
        device=args.device,
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads
    )
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        device=device,
        quantize=args.quantize
    )

    benchmarker = ModelBenchmarker(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    results = benchmarker.run_all_tests(model_name=args.model_name)

    results.print_summary()
    # 可选择打印详细结果
    # results.print_detailed()
    
    # 运行性能分析
    benchmarker.run_profiler(
        input_text="Hello world!", 
        max_new_tokens=100,
        output_file="profile_trace.json"
    )


if __name__ == "__main__":
    main()
    