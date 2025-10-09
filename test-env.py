import torch

if torch.cuda.is_available():
    print(f"avaiable GPU num: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("no GPU")

print("if ZLUDA:", "ZLuda" in torch.__version__ or "zlu" in torch.__config__.show())

print("if ROCm HIP:", hasattr(torch.version, "hip") and torch.version.hip is not None)

print("\nPyTorch config:")
print(torch.__config__.show())