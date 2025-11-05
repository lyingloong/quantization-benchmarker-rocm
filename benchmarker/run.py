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
    if not (quantize and quantize_method == "bitsandbytes"):
        # use no-cudagraphs to avoid cudagraphs related errors, especially for amd-quark
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
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
        elif quantize_method == "gptq-int8":
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
    
                    quant_config = QuantizeConfig(bits=8, group_size=128)
                    del model
                    model = GPTQModel.load(model_path, quant_config, device="cuda")
                    model.quantize(calibration_dataset, batch_size=1)
                    model.save("../qwen3_gptqmodel-8bit")
                    model.model.to("cuda")
                except ImportError:
                    print("[load_model_and_tokenizer] gptq not installed, skipping quantization.")      
            else:
                try:
                    model.tie_weights()
                except ImportError:
                    print("[load_model_and_tokenizer] gptq not installed, skipping loading.")
        elif quantize_method == "gptq-int4":
            if(not already_quantized):
                print("[load_model_and_tokenizer] Applying int4 quantization using gptq...")
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
                    model = GPTQModel.load(model_path, quant_config, device="cuda")
                    model.quantize(calibration_dataset, batch_size=1)
                    model.save("../qwen3_gptqmodel-4bit")
                    model.model.to("cuda")
                except ImportError:
                    print("[load_model_and_tokenizer] gptq not installed, skipping quantization.")      
            else:
                try:
                    model.tie_weights()
                except ImportError:
                    print("[load_model_and_tokenizer] gptq not installed, skipping loading.")
        elif quantize_method == "bitsandbytes":
            print("[load_model_and_tokenizer] Applying 8-bit quantization using bitsandbytes...")
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                del model  # 删除之前加载的 float16 模型
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",               # 自动分配到 GPU
                    attn_implementation="sdpa",
                    trust_remote_code=True
                )
            except ImportError:
                print("[load_model_and_tokenizer] bitsandbytes not installed, skipping quantization.")
        elif quantize_method.startswith("quark-"):
            calib_file = f"quark_{quantize_method}_calib.pt"
            
            try:
                from quark.torch import ModelQuantizer
                from quark.torch.quantization import (
                    Config, QuantizationConfig,
                    FP8E4M3PerTensorSpec, Int8PerTensorSpec, Int4PerTensorSpec
                )
                from datasets import load_dataset
                import os

                print(f"[load_model_and_tokenizer] Applying {quantize_method} quantization using Quark...")

                device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # ====== 定义量化规格 ======
                if quantize_method == "quark-fp8":
                    spec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
                elif quantize_method == "quark-int8":
                    spec = Int8PerTensorSpec(observer_method="min_max", is_dynamic=False, symmetric=True, round_method="round", scale_type="float").to_quantization_spec()
                elif quantize_method == "quark-int4":
                    spec = Int4PerTensorSpec(observer_method="min_max", is_dynamic=False, symmetric=True, round_method="round", scale_type="float").to_quantization_spec()
                else:
                    raise ValueError(f"Unsupported quark quantize method: {quantize_method}")

                # ====== 全局与 KV-cache 配置 ======
                global_quant_config = QuantizationConfig(input_tensors=spec, weight=spec)
                kv_cache_layer_names = ["*k_proj", "*v_proj"]
                kv_cache_quant_config = {
                    name: QuantizationConfig(
                        input_tensors=global_quant_config.input_tensors,
                        weight=global_quant_config.weight,
                        output_tensors=spec
                    )
                    for name in kv_cache_layer_names
                }
                exclude_layers = ["lm_head"]

                quant_config = Config(
                    global_quant_config=global_quant_config,
                    layer_quant_config=kv_cache_quant_config,
                    kv_cache_quant_config=kv_cache_quant_config,
                    exclude=exclude_layers
                )

                quantizer = ModelQuantizer(quant_config)

                if os.path.exists(calib_file):
                    # 加载校准结果
                    quantizer.load_calibration(calib_file)
                    model = quantizer.quantize_model(model, None)
                    print(f"[load_model_and_tokenizer] Loaded cached calibration from {calib_file}")
                else:
                    # ====== 校准数据 ======
                    dataset = load_dataset(
                        "allenai/c4",
                        data_files="en/c4-train.00001-of-01024.json.gz",
                        split="train"
                    ).select(range(32))
                    text_data = list(dataset["text"])

                    tokenized_outputs = tokenizer(
                        text_data,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    tokenized_outputs = {k: v.to(device) for k, v in tokenized_outputs.items()}

                    # ====== 生成 batch_size=1 的列表用于校准 ======
                    calib_dataloader = [
                        {k: v[i:i+1] for k, v in tokenized_outputs.items()}
                        for i in range(tokenized_outputs["input_ids"].size(0))
                    ]

                    # ====== 迭代 batch 做校准 ======
                    with torch.no_grad():
                        for batch in calib_dataloader:
                            model(**batch)

                    # ====== 执行量化 ======
                    model = quantizer.quantize_model(model, calib_dataloader)

                print(f"[load_model_and_tokenizer] Quark ({quantize_method}) quantization done.")
                print(f"[load_model_and_tokenizer] KV cache layers: {list(kv_cache_quant_config.keys())}")

            except Exception as e:
                print(f"[load_model_and_tokenizer] Quark quantization failed: {e}")
                raise e
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
        output_file=f"result/{args.model_name}_{args.quantize_method}.txt" if args.quantize else f"result/{args.model_name}_origin.txt"
    )


if __name__ == "__main__":
    main()
    
