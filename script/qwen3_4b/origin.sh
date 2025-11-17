MODEL_PATH="../QWEN3"
MODEL_NAME="Qwen3-4B"
DEVICE="cuda"

# export TORCH_PROFILER_DISABLE_FLOWS=1

python run.py \
   --model_name "$MODEL_NAME" \
   --model_path "$MODEL_PATH" \
   --max_new_tokens_list "1" \
   --device "$DEVICE" > qwen3-4b_origin_benchmark.log 2>&1

echo "[qwen3-4b/origin.sh] Done."