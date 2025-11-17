import time
import torch
from typing import Callable, Any, Optional, List
from .benchmark_config import BenchmarkConfig
from .benchmark_result import BenchmarkResultCollection, SingleTestResult
import os

class ModelBenchmarker:
    """通用模型性能测试器"""
    
    def __init__(self, 
                 model: Any, 
                 tokenizer: Any, 
                 config: BenchmarkConfig,
                 debug: bool = False,
                 generate_func: Optional[Callable] = None):
        """
        初始化性能测试器
        
        Args:
            model: 待测试的模型
            tokenizer: 对应的tokenizer
            config: 测试配置
            generate_func: 自定义生成函数，默认为model.generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.debug = debug
        self.generate_func = generate_func or self._default_generate_func
        
        # 配置设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"[ModelBenchmarker: __init__] Device: {self.device}")
        
        # set threads
        try:
            if config.num_threads is not None:
                torch.set_num_threads(config.num_threads)
                print(f"[ModelBenchmarker: __init__] Set num_threads to {config.num_threads}")
            if config.num_interop_threads is not None:
                torch.set_num_interop_threads(config.num_interop_threads)
                print(f"[ModelBenchmarker: __init__] Set num_interop_threads to {config.num_interop_threads}")
        except RuntimeError as e:
            if "cannot set number of interop threads" in str(e) or "cannot set number of threads" in str(e):
                print(f"[ModelBenchmarker: __init__] Warning: {e}")

        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 结果收集
        self.results = BenchmarkResultCollection()
    
    def _default_generate_func(self, model, inputs, **kwargs):
        """默认的生成函数"""
        return model.generate(** inputs, **kwargs)
    
    def _prepare_inputs(self, input_text: str) -> dict:
        """准备输入数据"""
        inputs = self.tokenizer(input_text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def run_warmup(self, input_text: str, max_new_tokens: int):
        """执行预热运行"""
        inputs = self._prepare_inputs(input_text)
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                self.generate_func(
                    self.model, 
                    inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )

    @torch.inference_mode()
    def run_single_test(self, 
                       input_text: str, 
                       max_new_tokens: int, 
                       config_name: str) -> SingleTestResult:
        """执行单次配置的测试"""
        inputs = self._prepare_inputs(input_text)
        input_length = inputs["input_ids"].shape[1]
        
        # 预热
        self.run_warmup(input_text, max_new_tokens)
        
        # 存储每次运行的指标
        latencies = []
        token_counts = []
        gpu_mem_usage = []
        
        # 多次运行
        for i in range(self.config.num_runs):
            # 记录GPU内存
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()  # 重置峰值统计

            # 记录时间
            start_time = time.perf_counter()
            
            # 推理
            with torch.no_grad():
                outputs = self.generate_func(
                    self.model, 
                    inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # 计算时间
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            # 计算内存使用
            mem_usage = 0.0
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated()
                mem_usage = peak_mem / (1024 ** 3)  # 转换为GB单位

            # 计算生成的token数量
            generated_tokens = outputs.shape[1] - input_length
            
            # 保存结果
            latencies.append(latency)
            token_counts.append(generated_tokens)
            gpu_mem_usage.append(mem_usage)

            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n--- 模型输出 ({i+1}/{self.config.num_runs}) ---\n{decoded_output}\n")

            print(f"  运行 {i+1}/{self.config.num_runs}: "
                  f"延迟={latency:.4f}s, 生成{generated_tokens} tokens, "
                  f"吞吐量={generated_tokens/latency:.2f} t/s, "
                  f"显存={mem_usage:.2f}GB")

        # 创建结果对象
        return SingleTestResult(
            config_name=config_name,
            input_length=input_length,
            max_new_tokens=max_new_tokens,
            latencies=latencies,
            token_counts=token_counts,
            gpu_mem_usage=gpu_mem_usage
        )
    
    def run_all_tests(self, model_name: str):
        """运行所有配置的测试"""
        print(f"\n=== 开始测试: {model_name} ===")
        
        for input_text in self.config.input_texts:
            # 为不同输入创建标识
            input_id = f"输入长度_{len(self.tokenizer(input_text)['input_ids'])}"
            
            for max_new_tokens in self.config.max_new_tokens_list:
                config_name = f"{model_name}_{input_id}_生成_{max_new_tokens}"
                print(f"\n--- 测试配置: {config_name} ---")
                
                # 运行测试
                result = self.run_single_test(
                    input_text=input_text,
                    max_new_tokens=max_new_tokens,
                    config_name=config_name
                )
                
                # 保存结果
                self.results.add_result(result)
        
        return self.results
    
    def run_profiler(self, input_text: str, max_new_tokens: int, output_file: str = "output_file.txt", trace_file: str = "trace_profiler.json"):
        """运行性能分析器"""
        try:
            from torch.profiler import profile, record_function, ProfilerActivity
        except ImportError:
            print("警告: 未安装torch.profiler，无法运行性能分析")
            return
        
        inputs = self._prepare_inputs(input_text)
        
        # 先预热
        self.run_warmup(input_text, max_new_tokens)
        
        print("\n=== 运行性能分析 ===")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                self.generate_func(
                    self.model, 
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
        
        # 生成表格字符串
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
        # print(table)
        
        with open(output_file, "w") as f:
            f.write(table)
        
        print(f"\n性能分析表格已保存至: {output_file}")

        if(self.debug):
            prof.export_chrome_trace(trace_file)
            print(f"\nTrace 已保存至: {trace_file}")