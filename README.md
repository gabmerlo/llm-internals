# llm-internals

LLM inference from scratch in PyTorch, to learn how it works before optimizing parts of it with my own CUDA kernels ([cuda-kernels](https://github.com/gabmerlo/cuda-kernels)). I only use Hugging Face for the tokenizer and the pretrained weights, and I check every piece against their model.

## GPT-2

- Embeddings, LayerNorm, MLP (GELU), multi-head attention, the 12 blocks and the LM head
- Logits match HF with a max diff of 8.39e-05 (4 sequences of 200 random tokens)
- Greedy generation
- KV cache, allocated once at max length with `head_dim` as the last dimension
- decode_attention: attention for a single decoding step, separate from the full attention
- GPU version for measurements

## Llama 3.2 1B

Built on top of my GPT-2 code:

- RMSNorm, SwiGLU, no biases
- RoPE with Llama 3 frequency scaling (thresholds from `config.rope_parameters`)
- GQA: 32 query heads, 8 KV heads, `repeat_interleave` in both attention and `decode_attention`
- KV cache sized to fit on a T4
- Logits match HF with a max diff of 8.58e-06

Bugs I hit along the way: GELU without the tanh approximation GPT-2 uses, a KV cache that I copied every step before allocating it once, max_len as the last cache dimension, reading V with K's cache index in decode_attention, and a wrong atol/rtol that failed correct results.

## Doing soon

- Measure the KV cache speedup (tokens/s)
- Read rope_theta and head_dim from the config instead of hardcoding them
