MODEL_PATH="../QWEN3"
MODEL_NAME="Qwen3-4B"
DEVICE="cuda"

QUANTIZED_MODEL="../qwen3_gptqmodel-8bit"

# export TORCH_PROFILER_DISABLE_FLOWS=1
    
# enable cache dequantized weights when inference with GPTQ models
export GPTQ_TORCH_CACHE_WEIGHTS=1

python run.py \
    --model_name "$MODEL_NAME" \
    --model_path "$QUANTIZED_MODEL" \
    --device "$DEVICE" \
    --max_new_tokens_list "1" \
    --debug \
    --quantize \
    --quantize_method "gptq-int8" \
    --already_quantized > qwen3-4b_gptq-int8_benchmark.log 2>&1

# QUANTIZED_MODEL="../qwen3_gptqmodel-4bit"

# python run.py \
#     --model_name "$MODEL_NAME" \
#     --model_path "$QUANTIZED_MODEL" \
#     --device "$DEVICE" \
#     --quantize \
#     --quantize_method "gptq-int4" \
#     --already_quantized > qwen3-4b_gptq-int4_benchmark.log 2>&1

echo "[qwen3-4b/gptq.sh] Done."