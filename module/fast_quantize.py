# module/fast_quantize.py
# only implement int8 per-channel quantization
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 量化函数 ======
def quantize_int8_per_channel(weight: torch.Tensor):
    """int8 per-channel symmetric quantization"""
    w = weight.float()
    max_val = w.abs().amax(dim=1, keepdim=True)
    scale = max_val / 127.0
    scale[scale == 0] = 1e-8
    w_q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return w_q, scale

# def quantize_int8_groupwise(weight: torch.Tensor, group_size: int = 128):
#     """int8 per-channel groupwise quantization (panel-wise)"""
#     w = weight.float()
#     n_groups = (w.shape[1] + group_size - 1) // group_size
#     scales = []
#     w_q = torch.zeros_like(w, dtype=torch.int8)
#     for g in range(n_groups):
#         start = g * group_size
#         end = min((g + 1) * group_size, w.shape[1])
#         w_group = w[:, start:end]
#         max_val = w_group.abs().amax(dim=1, keepdim=True)
#         scale = max_val / 127.0
#         scale[scale == 0] = 1e-8
#         w_q[:, start:end] = torch.round(w_group / scale).clamp(-127, 127).to(torch.int8)
#         scales.append(scale)
#     scales = torch.cat(scales, dim=1)
#     return w_q, scales

# def quantize_int4_per_channel(weight: torch.Tensor):
#     """int4 per-channel symmetric quantization"""
#     w = weight.float()
#     max_val = w.abs().amax(dim=1, keepdim=True)
#     scale = max_val / 7.0
#     scale[scale == 0] = 1e-8
#     w_q = torch.round(w / scale).clamp(-7, 7).to(torch.int8)
#     return w_q, scale

# def quantize_int4_groupwise(weight: torch.Tensor, group_size: int = 128):
#     """int4 per-channel groupwise quantization"""
#     w = weight.float()
#     n_groups = (w.shape[1] + group_size - 1) // group_size
#     scales = []
#     zeros = []
#     w_q = torch.zeros_like(w, dtype=torch.int8)
#     for g in range(n_groups):
#         start = g * group_size
#         end = min((g + 1) * group_size, w.shape[1])
#         w_group = w[:, start:end]
#         min_val, max_val = w_group.min(dim=1, keepdim=True)[0], w_group.max(dim=1, keepdim=True)[0]
#         scale = (max_val - min_val) / 15.0
#         scale[scale == 0] = 1e-8
#         zero = min_val
#         w_q[:, start:end] = torch.round((w_group - zero) / scale).clamp(0, 15).to(torch.int8)
#         scales.append(scale)
#         zeros.append(zero)
#     scales = torch.cat(scales, dim=1)
#     zeros = torch.cat(zeros, dim=1)
#     return w_q, scales, zeros

# ====== 反量化函数 ======
def dequantize_int8(weight_q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32):
    return (weight_q.to(dtype) * scale.to(dtype))

def dequantize_int4(weight_q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor = None, dtype: torch.dtype = torch.float32):
    weight_q = weight_q.to(dtype)
    scale = scale.to(dtype)
    if zero is None:
        return weight_q * scale
    else:
        zero = zero.to(dtype)
        return weight_q * scale + zero

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mode="int8_per_channel", group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.group_size = group_size
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # 量化权重和参数（不保存原 float 权重）
        self.register_buffer("weight_q", None)
        self.register_buffer("scale", None)
        self.register_buffer("zero", None)  # int4 groupwise 时用

    @torch.no_grad()
    def quantize_weight(self, weight: torch.Tensor):
        if self.mode == "int8_per_channel":
            self.weight_q, self.scale = quantize_int8_per_channel(weight)
            self.zero = None
        elif self.mode == "int8_groupwise":
            self.weight_q, self.scale = quantize_int8_groupwise(weight, self.group_size)
            self.zero = None
        elif self.mode == "int4_per_channel":
            self.weight_q, self.scale = quantize_int4_per_channel(weight)
            self.zero = None
        elif self.mode == "int4_groupwise":
            self.weight_q, self.scale, self.zero = quantize_int4_groupwise(weight, self.group_size)
        else:
            raise ValueError(f"Unsupported quantization mode: {self.mode}")

    def forward(self, x):
        # forward 只使用量化权重和量化参数
        if self.weight_q is None:
            raise RuntimeError("Weight has not been quantized. Call quantize_weight first.")
        if self.mode.startswith("int8"):
            weight_f = dequantize_int8(self.weight_q, self.scale, dtype=x.dtype)
        else:
            weight_f = dequantize_int4(self.weight_q, self.scale, self.zero, dtype=x.dtype)
        return F.linear(x, weight_f, self.bias)

'''
defined subclasses for optimization
'''
class QuantLinearInt8PerChannel(QuantLinear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias, mode="int8_per_channel")
    
    def forward(self, x):
        if self.weight_q is None:
            raise RuntimeError("Weight has not been quantized. Call quantize_weight first.")
        weight_f = dequantize_int8(self.weight_q, self.scale, dtype=x.dtype)
        return F.linear(x, weight_f, self.bias)

class QuantLinearInt8Groupwise(QuantLinear):
    def __init__(self, in_features, out_features, bias=True, group_size=128):
        super().__init__(in_features, out_features, bias, mode="int8_groupwise", group_size=group_size)
    
    def forward(self, x):
        if self.weight_q is None:
            raise RuntimeError("Weight has not been quantized. Call quantize_weight first.")
        weight_f = dequantize_int8(self.weight_q, self.scale, dtype=x.dtype)
        return F.linear(x, weight_f, self.bias)

class QuantLinearInt4PerChannel(QuantLinear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias, mode="int4_per_channel")
    
    def forward(self, x):
        if self.weight_q is None:
            raise RuntimeError("Weight has not been quantized. Call quantize_weight first.")
        weight_f = dequantize_int4(self.weight_q, self.scale, dtype=x.dtype)
        return F.linear(x, weight_f, self.bias)

class QuantLinearInt4Groupwise(QuantLinear):
    def __init__(self, in_features, out_features, bias=True, group_size=128):
        super().__init__(in_features, out_features, bias, mode="int4_groupwise", group_size=group_size)
    
    def forward(self, x):
        if self.weight_q is None:
            raise RuntimeError("Weight has not been quantized. Call quantize_weight first.")
        weight_f = dequantize_int4(self.weight_q, self.scale, self.zero, dtype=x.dtype)
        return F.linear(x, weight_f, self.bias)


def replace_linear(model: nn.Module, mode="int8_per_channel", group_size=128, exclude=None):
    """
    递归替换 nn.Linear 为 QuantLinear
    mode: int8_per_channel / int8_groupwise / int4_per_channel / int4_groupwise
    """
    if exclude is None:
        exclude = []

    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if any([name.endswith(ex) for ex in exclude]):
                continue
            match mode:
                case "int8_per_channel":
                    qlinear = QuantLinearInt8PerChannel(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None
                    )
                case "int8_groupwise":
                    qlinear = QuantLinearInt8Groupwise(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        group_size=group_size
                    )
                case "int4_per_channel":
                    qlinear = QuantLinearInt4PerChannel(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None
                    )
                case "int4_groupwise":
                    qlinear = QuantLinearInt4Groupwise(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        group_size=group_size
                    )
                case _:
                    raise ValueError(f"Unsupported quantization mode: {mode}")
            qlinear.quantize_weight(child.weight.data)
            if child.bias is not None:
                qlinear.bias.data.copy_(child.bias.data)
            setattr(model, name, qlinear)
        else:
            replace_linear(child, mode, group_size, exclude)


