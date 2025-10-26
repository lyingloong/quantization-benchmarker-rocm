MODEL_PATH="../QWEN3"
MODEL_NAME="Qwen3-4B"
QUANTIZED_MODEL="../qwen3_gptqmodel-4bit"
DEVICE="cuda"

# export TORCH_PROFILER_DISABLE_FLOWS=1

# python benchmarker/run.py \
#    --model_name "$MODEL_NAME" \
#    --model_path "$MODEL_PATH" \
#    --device "$DEVICE"

# python benchmarker/run.py \
#    --model_name "$MODEL_NAME" \
#    --model_path "$MODEL_PATH" \
#    --device "$DEVICE" \
#    --quantize \
#    --quantize_method "torchao-int8"
    
# python benchmarker/run.py \
#     --model_name "$MODEL_NAME" \
#     --model_path "$QUANTIZED_MODEL" \
#     --device "$DEVICE" \
#     --quantize \
#     --quantize_method "gptq" \
#     --already_quantized

python benchmarker/run.py \
   --model_name "$MODEL_NAME" \
   --model_path "$MODEL_PATH" \
   --device "$DEVICE" \
   --quantize \
   --quantize_method "bitsandbytes"

echo "[qwen3-4b.sh] Done."