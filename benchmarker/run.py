from benchmark_config import BenchmarkConfig
from model_benchmarker import ModelBenchmarker
from transformers import AutoModelForCausalLM,AutoTokenizer
from tokenizers import Tokenizer
import torch
import argparse
import os

# 关闭 triton/exllama 内核（ROCm 下无效反而拖慢）
os.environ["DISABLE_TRITON"] = "1"
os.environ["DISABLE_EXLLAMA"] = "1"

# 降低 CPU 内核调度干扰
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_and_tokenizer(model_path: str, device: str, quantize: bool = False, quantize_method: str = "torchao-int8", already_quantized : bool = False):
    print("[load_model_and_tokenizer] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  
        device_map="cuda",
        attn_implementation="sdpa",
        dtype=torch.float16,
        trust_remote_code=True
    )
    model = torch.compile(model, mode="reduce-overhead")
    print(next(model.parameters()).device)
    print(model.hf_device_map)
    for name, param in model.named_parameters():
        assert param.device.type == "cuda", f"{name} still on CPU"
    if quantize:
        if quantize_method == "torchao-int8":
            print("[load_model_and_tokenizer] Applying int8 quantization using torchao...")
            try:
                from torchao.quantization import quantize_, Int8WeightOnlyConfig
                quantize_(model, Int8WeightOnlyConfig())
                print("[load_model_and_tokenizer] Model quantized to int8.")
            except ImportError:
                print("[load_model_and_tokenizer] torchao not installed, skipping quantization.")
        elif quantize_method == "gptq":
            if(not already_quantized):
                print("[load_model_and_tokenizer] Applying int8 quantization using gptq...")
                try:
                    from gptqmodel import GPTQModel,QuantizeConfig
                    from datasets import load_dataset
                    calibration_dataset = load_dataset(
                        "allenai/c4",
                        data_files="en/c4-train.00001-of-01024.json.gz",
                        split="train"
                      ).select(range(1024))["text"]
    
                    quant_config = QuantizeConfig(bits=4, group_size=128)
                    del model
                    model = GPTQModel.load(model_path, quant_config)
                    model.tokenizer =tokenizer
                    model.quantize(calibration_dataset, batch_size=1)
                except ImportError:
                    print("[load_model_and_tokenizer] gptq not installed, skipping quantization.")      
            else:
                try:
                    model.tie_weights()
                except ImportError:
                    print("[load_model_and_tokenizer] gptq not installed, skipping loading.")
        else:
            raise ValueError(f"Unsupported quantization method: {quantize_method}")
    
    return model, tokenizer

def main():
    # 设置随机种子
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="run benchmark")
    parser.add_argument("--model_name", type=str, default="Model", help="name of the model")
    parser.add_argument("--model_path", type=str, required=True, help="path of the model")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the model")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize the model")
    parser.add_argument("--already_quantized", action="store_true", help="whether to load quantized the model")
    parser.add_argument("--quantize_method", type=str, default="torchao-int8", help="quantization method")
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
        quantize=args.quantize,
        quantize_method=args.quantize_method,
        already_quantized=args.already_quantized
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
    