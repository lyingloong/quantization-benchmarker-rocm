MODEL_PATH="../QWEN3"
MODEL_NAME="Qwen3-4B"
DEVICE="cuda"

# export TORCH_PROFILER_DISABLE_FLOWS=1

python run.py \
   --model_name "$MODEL_NAME" \
   --model_path "$MODEL_PATH" \
   --device "$DEVICE" \
   --max_new_tokens_list "1" \
   --quantize \
   --quantize_method "fast-int8-perchannel" > qwen3-4b_fast-int8-perchannel_benchmark.log 2>&1

# python run.py \
#    --model_name "$MODEL_NAME" \
#    --model_path "$MODEL_PATH" \
#    --device "$DEVICE" \
#    --quantize \
#    --quantize_method "fast-int8-groupwise" > qwen3-4b_fast-int8-groupwise_benchmark.log 2>&1
    
echo "[qwen3-4b/fast.sh] Done."