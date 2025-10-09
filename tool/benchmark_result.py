from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class SingleTestResult:
    """单次测试结果"""
    config_name: str
    input_length: int
    max_new_tokens: int
    latencies: List[float]          # 每次运行的延迟(秒)
    token_counts: List[int]         # 每次生成的token数量
    gpu_mem_usage: List[float]      # 每次运行的GPU内存使用(GB)
    
    @property
    def avg_latency(self) -> float:
        """平均延迟"""
        return np.mean(self.latencies)
    
    @property
    def std_latency(self) -> float:
        """延迟标准差"""
        return np.std(self.latencies)
    
    @property
    def avg_throughput(self) -> float:
        """平均吞吐量(tokens/sec)"""
        return np.mean([t/c for t, c in zip(self.token_counts, self.latencies)])
    
    @property
    def avg_tokens(self) -> float:
        """平均生成token数"""
        return np.mean(self.token_counts)
    
    @property
    def avg_gpu_mem(self) -> float:
        """平均GPU内存使用"""
        return np.mean(self.gpu_mem_usage)
    
    @property
    def peak_gpu_mem(self) -> float:
        """峰值GPU内存使用"""
        return np.max(self.gpu_mem_usage)


class BenchmarkResultCollection:
    """测试结果集合，管理多个测试结果"""
    def __init__(self):
        self.results: Dict[str, SingleTestResult] = {}
    
    def add_result(self, result: SingleTestResult):
        """添加测试结果"""
        self.results[result.config_name] = result
    
    def get_result(self, config_name: str) -> Optional[SingleTestResult]:
        """获取指定配置的测试结果"""
        return self.results.get(config_name)
    
    def print_summary(self):
        """打印所有测试结果摘要"""
        print("\n" + "="*80)
        print(f"性能测试汇总 ({len(self.results)} 种配置)")
        print("="*80)
        print(f"{'配置名称':<30} | {'平均延迟(s)':<12} | {'吞吐量(t/s)':<12} | {'峰值显存(GB)':<12}")
        print("-"*80)
        
        for result in self.results.values():
            print(f"{result.config_name[:27]}... | {result.avg_latency:.4f}      | {result.avg_throughput:.2f}       | {result.peak_gpu_mem:.2f}")
        
        print("="*80 + "\n")
    
    def print_detailed(self, config_name: Optional[str] = None):
        """打印详细测试结果"""
        if config_name:
            results = [self.get_result(config_name)] if self.get_result(config_name) else []
        else:
            results = self.results.values()
        
        for result in results:
            print("\n" + "="*60)
            print(f"详细结果 - {result.config_name}")
            print("="*60)
            print(f"输入长度: {result.input_length} tokens")
            print(f"生成设置: 最多 {result.max_new_tokens} tokens")
            print("-"*60)
            print(f"平均延迟: {result.avg_latency:.4f}s (±{result.std_latency:.4f}s)")
            print(f"平均吞吐量: {result.avg_throughput:.2f} tokens/sec")
            print(f"平均生成token数: {result.avg_tokens:.1f}")
            print(f"平均GPU显存使用: {result.avg_gpu_mem:.2f}GB")
            print(f"峰值GPU显存使用: {result.peak_gpu_mem:.2f}GB")
            print("="*60 + "\n")
    
    def to_dict(self) -> Dict:
        """转换为字典格式，便于保存"""
        return {
            name: {
                "config_name": result.config_name,
                "input_length": result.input_length,
                "max_new_tokens": result.max_new_tokens,
                "avg_latency": result.avg_latency,
                "std_latency": result.std_latency,
                "avg_throughput": result.avg_throughput,
                "avg_tokens": result.avg_tokens,
                "avg_gpu_mem": result.avg_gpu_mem,
                "peak_gpu_mem": result.peak_gpu_mem,
                "raw_data": {
                    "latencies": result.latencies,
                    "token_counts": result.token_counts,
                    "gpu_mem_usage": result.gpu_mem_usage
                }
            }
            for name, result in self.results.items()
        }
    