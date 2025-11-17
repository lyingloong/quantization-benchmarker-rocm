from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BenchmarkConfig:
    # 测试运行参数
    num_runs: int = 10                  # 正式测试次数
    warmup_runs: int = 3                # 预热次数
    max_new_tokens_list: List[int] = None  # 生成token长度列表
    input_texts: List[str] = None       # 测试输入文本列表
    
    # 设备配置
    device: str = "cuda"                # 运行设备
    num_threads: Optional[int] = None   # CPU线程数
    num_interop_threads: Optional[int] = None  # 算子间并行线程数
    
    """
    推理参数
    do_sample: if True, use sampling; otherwise, use greedy decoding.
    temperature: sampling temperature; higher values mean more random results.
    top_p: if set to < 1, only the most probable tokens with probabilities that
           add up to top_p or higher are kept for generation.
    """
    do_sample: bool = False             # 是否采样生成
    temperature: float = 1.0            # 采样温度
    top_p: float = 1.0                  # top_p参数

    debug: bool = False
    
    def __post_init__(self):
        # 设置默认值
        if self.max_new_tokens_list is None:
            if self.debug:
                self.max_new_tokens_list = [1]
            else:
                self.max_new_tokens_list = [1, 50, 100]
        
        if self.input_texts is None:
            if self.debug:
                self.input_texts = ["Hello"]
            else:
                self.input_texts = [
                    "Hello world!",  # 短输入
                    "Please provide a detailed explanation of artificial intelligence "
                    "and its applications in modern society, including examples from "
                    "healthcare, finance, and transportation sectors."  # 长输入
                ]
    