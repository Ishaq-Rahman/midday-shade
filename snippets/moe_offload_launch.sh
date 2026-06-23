#!/usr/bin/env bash
# The launch that fits a 35B Mixture-of-Experts model on a 12 GB GPU.
#
# The trick is NOT "shrink the model until it fits in VRAM". A MoE model activates
# only ~3B of its 35B parameters per token, so the winning split is:
#
#   --n-gpu-layers 99   every layer's ATTENTION + the KV cache on the GPU
#                       (attention is what gets expensive at long context, so it
#                        earns the scarce VRAM)
#   --n-cpu-moe N       the big per-expert FFN tensors stay in system RAM; only
#                       ~3B are touched per token, so the CPU keeps up
#   --flash-attn on     rocWMMA flash-attention (llama.cpp built from source for
#                       AMD ROCm/HIP) makes a full 64K context fit
#   --cache-type-k/v q8_0   q8_0 KV cache beats q4_0 at depth — measured, not assumed
#   --mlock             pin the model in RAM so MoE experts never page to swap
#                       (generation is CPU-expert-bound; a swapped expert stalls)
#   --threads 6         the measured optimum on this CPU, not the core count
#
# Every knob here was chosen from benchmark data, not defaults. Result:
# ~32 tokens/s, sustained 26 t/s at 50K context. See docs/llm-infra.md.

exec "$LLAMA_BIN" \
  --model "$MODEL" \
  --no-mmap \
  --mlock \
  --n-gpu-layers 99 \
  --n-cpu-moe "$N_CPU_MOE" \
  --ctx-size "$CTX" \
  --threads "$THREADS" \
  --parallel 1 \
  --cont-batching \
  --flash-attn on \
  --cache-type-k "$CTK" --cache-type-v "$CTV" \
  --batch-size "$BATCH" --ubatch-size "$UBATCH" \
  --jinja \
  --alias "${ALIAS:-tuesday}" \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --host "$HOST" --port "$PORT"
