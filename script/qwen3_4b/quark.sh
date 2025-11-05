MODEL_PATH="../QWEN3"
MODEL_NAME="Qwen3-4B"
DEVICE="cuda"

# export TORCH_PROFILER_DISABLE_FLOWS=1
    
# python benchmarker/run.py \
#     --model_name "$MODEL_NAME" \
#     --model_path "$MODEL_PATH" \
#     --device "$DEVICE" \
#     --quantize \
#     --quantize_method "quark-fp8" > qwen3-4b_quark-fp8_benchmark.log 2>&1

python benchmarker/run.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --quantize \
    --quantize_method "quark-int8" > qwen3-4b_quark-int8_benchmark.log 2>&1

python benchmarker/run.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --quantize \
    --quantize_method "quark-int4" > qwen3-4b_quark-int4_benchmark.log 2>&1

echo "[qwen3-4b/quark.sh] Done."