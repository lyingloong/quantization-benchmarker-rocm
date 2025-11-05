MODEL_PATH="../QWEN3"
MODEL_NAME="Qwen3-4B"
DEVICE="cuda"

# export TORCH_PROFILER_DISABLE_FLOWS=1

python benchmarker/run.py \
   --model_name "$MODEL_NAME" \
   --model_path "$MODEL_PATH" \
   --device "$DEVICE" \
   --quantize \
   --quantize_method "bitsandbytes" > qwen3-4b_bnb_benchmark.log 2>&1

echo "[qwen3-4b/bnb.sh] Done."