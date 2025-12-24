# Nanochat Model Comparison

The **architecture is exactly the same** - it's all the same Transformer design. The only differences are **scale parameters**:

| Aspect | CPU/MPS Demo (d4) | $100 GPU (d20) | $1000 GPU (d32) |
|--------|-------------------|----------------|-----------------|
| **Layers (depth)** | 4 | 20 | 32 |
| **Model dim** | 256 | ~1024 | 2048 |
| **Heads** | 2 | ~8 | 16 |
| **Parameters** | ~36.7M | ~561M | ~1.9B |
| **Training tokens** | ~10K | ~11.2B | ~37.6B |
| **Training time** | ~3 min | ~4 hrs | ~31 hrs |
| **Cost** | $0 | ~$100 | ~$800 |
| **GPUs** | 0 (MPS/CPU) | 8×H100 | 8×H100 |

## Key Point

The fundamental architecture (RoPE, GQA, RMSNorm, ReLU², etc.) is identical across all models. You just scale up:
- More layers (depth)
- Wider layers (n_embd)
- More attention heads
- More training data & compute

This is the essence of "scaling laws" - the same architecture, just bigger.
