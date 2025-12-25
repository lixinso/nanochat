# Nanochat Training Pipeline

The model is trained in **two stages** using the same architecture but different data:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NANOCHAT TRAINING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

    Stage 1: PRETRAINING                    Stage 2: SFT (Fine-tuning)
    ────────────────────                    ──────────────────────────
    
    ┌─────────────────────┐                 ┌─────────────────────┐
    │  Random Init Model  │                 │  Pretrained Model   │
    │  (knows nothing)    │                 │  (knows language)   │
    └──────────┬──────────┘                 └──────────┬──────────┘
               │                                       │
               ▼                                       ▼
    ┌─────────────────────┐                 ┌─────────────────────┐
    │  FineWeb-Edu-100B   │                 │  SmolTalk + Tasks   │
    │  (raw text)         │                 │  (conversations)    │
    └──────────┬──────────┘                 └──────────┬──────────┘
               │                                       │
               ▼                                       ▼
    ┌─────────────────────┐                 ┌─────────────────────┐
    │  base_train.py      │ ──────────────► │  chat_sft.py        │
    │  Learn language     │   checkpoint    │  Learn to chat      │
    └─────────────────────┘                 └─────────────────────┘
               │                                       │
               ▼                                       ▼
         "Base Model"                           "Chat Model"
       (text completion)                    (instruction following)
```

## Stage 1: Base Pretraining

| Aspect | Details |
|--------|---------|
| **Script** | `scripts/base_train.py` |
| **Data** | FineWeb-Edu-100B (~100B tokens, ~400-500 GB) |
| **Format** | Raw text documents |
| **Source** | `huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle` |
| **Storage** | 1,823 parquet files, downloaded on-demand to `~/.nanochat/base_data/` |
| **Purpose** | Teach language understanding, facts, and reasoning |
| **Loss** | All tokens are predicted (standard language modeling) |

**Data flow:**
```
Parquet files → raw text → tokenize → [token1, token2, token3, ...]
Input:  [token1, token2, token3, ...]
Target: [token2, token3, token4, ...]  (shifted by 1)
```

## Stage 2: Supervised Fine-Tuning (SFT)

| Aspect | Details |
|--------|---------|
| **Script** | `scripts/chat_sft.py` |
| **Data** | SmolTalk (460K train + 24K test conversations) |
| **Format** | Multi-turn conversations with user/assistant roles |
| **Source** | `huggingface.co/datasets/HuggingFaceTB/smol-smoltalk` |
| **Purpose** | Teach instruction-following and chat behavior |
| **Loss** | Only assistant responses (masked loss) |

**Data flow with masking:**
```
User: What is 2+2?     ← mask=0 (don't learn from this)
Assistant: 4           ← mask=1 (learn this!)
```

## Key Differences

| Aspect | Base Pretraining | Chat SFT |
|--------|------------------|----------|
| **Data format** | Raw text stream | Conversations with roles |
| **Tokenization** | Simple `encode()` | `render_conversation()` with mask |
| **Loss mask** | All tokens | Only assistant responses |
| **Padding** | None (continuous) | Padded to max length in batch |
| **Target for masked** | N/A | `-1` (PyTorch ignore index) |
| **Data size** | ~100B tokens | ~460K conversations |

## Evaluation Tasks

After training, models are evaluated on:

| Task | Dataset | Purpose |
|------|---------|---------|
| **MMLU** | cais/mmlu | Knowledge & reasoning |
| **GSM8K** | openai/gsm8k | Math word problems |
| **HumanEval** | openai/openai_humaneval | Code generation |
| **ARC** | allenai/ai2_arc | Science reasoning |

## Summary

**Same architecture, same weights** — SFT continues training from the pretrained checkpoint:

1. **Pretrain** on FineWeb → saves checkpoint (e.g., `base_d20.pt`)
2. **SFT** loads that checkpoint → fine-tunes on conversations → saves chat model (e.g., `chat_d20.pt`)

This is the standard LLM training approach (GPT, LLaMA, etc.): **pretrain for knowledge, then fine-tune for behavior**.
