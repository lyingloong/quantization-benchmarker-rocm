MODEL_PATH="../QWEN3"
MODEL_NAME="Qwen3-4B"
DEVICE="cuda"

# export TORCH_PROFILER_DISABLE_FLOWS=1

python benchmarker/run.py \
   --model_name "$MODEL_NAME" \
   --model_path "$MODEL_PATH" \
   --device "$DEVICE" > qwen3-4b_origin_benchmark.log 2>&1

# python benchmarker/run.py \
#    --model_name "$MODEL_NAME" \
#    --model_path "$MODEL_PATH" \
#    --device "$DEVICE" \
#    --use_vllm  > qwen3-4b_vllm_origin_benchmark.log 2>&1

echo "[qwen3-4b/origin.sh] Done."