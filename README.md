# MiniRobust-LM

**Adversarially robust, parameter-efficient, and compressed LLM
fine-tuned from Mistral-1B**

![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-blue)

## Overview
MiniRobust-LM is an end-to-end research pipeline addressing three 
core challenges in production LLM deployment:

- **Robustness** — hardening models against adversarial prompt 
injection and jailbreaks using PGD and TextFooler attacks
- **Efficiency** — parameter-efficient fine-tuning of Mistral-1B 
using LoRA on consumer-grade hardware
- **Compression** — GPTQ 4-bit quantization targeting <500MB 
deployment size with <5% accuracy degradation on MMLU

## Architecture
MiniRobust-LM/
├── train/
│ ├── train_classifier.py # ResNet-18 on CIFAR-10 w/ PGD
│ └── finetune_llm.py # Mistral-1B LoRA fine-tuning
├── attack/
│ ├── pgd_attack.py # PGD adversarial attack
│ └── textfooler_attack.py # TextFooler NLP attack
├── compress/
│ └── quantize.py # GPTQ 4-bit quantization
├── eval/
│ └── benchmark.py # MMLU + robustness benchmarking
└── notebooks/
└── demo.ipynb # End-to-end demo

## Modules
### 1. Adversarial Image Classification
Training ResNet-18 from scratch on CIFAR-10 with PGD adversarial 
training. Measures clean accuracy vs robust accuracy tradeoff under 
varying attack strengths (ε = 8/255).

### 2. Parameter-Efficient LLM Fine-tuning
Fine-tuning Mistral-1B on OpenHermes 2.5 instruction dataset using 
LoRA (rank=16, alpha=32) via HuggingFace PEFT. Optimized for 
single T4 GPU training with gradient checkpointing and mixed 
precision.

### 3. Adversarial Prompt Hardening
Generating adversarial prompt attacks using TextFooler and custom 
PGD-based prompt injection suite. Model is retrained on augmented 
adversarial dataset to reduce attack success rate.

### 4. Model Compression
Applying GPTQ 4-bit post-training quantization using auto-gptq. 
Benchmarking size, latency, and MMLU accuracy across baseline, 
quantized, and pruned variants.

### 5. Training Diagnostics
Full experiment tracking via Weights & Biases. Documenting and 
resolving training instabilities including loss spikes (gradient 
clipping), overfitting (augmentation + dropout), and slow 
convergence (cosine LR scheduling).

## Results
> Updated as experiments complete
| Metric | Baseline | Hardened | Compressed |
|--------|----------|----------|------------|
| MMLU Score | — | — | — |
| Attack Success Rate | — | — | — |
| Clean Accuracy (CIFAR-10) | — | — | — |
| Robust Accuracy (PGD) | — | — | — |
| Model Size | ~6 GB | ~6 GB | <500 MB |
| Inference Speed | — | — | — |

## Tech Stack
| | |
|---|---|
| Framework | PyTorch 2.x |
| Fine-tuning | HuggingFace Transformers, PEFT |
| Compression | bitsandbytes, auto-gptq |
| Attacks | TextAttack, custom PGD |
| Evaluation | lm-evaluation-harness |
| Tracking | Weights & Biases |
| Compute | Google Colab T4 |

## Setup
```bash
git clone https://github.com/avanikumar/MiniRobust-LM.git
cd MiniRobust-LM

pip install torch transformers peft bitsandbytes
pip install textattack auto-gptq wandb
pip install lm-eval datasets accelerate
```

## References
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)
- [Madry et al. — Adversarial Robustness via PGD](https://arxiv.org/abs/1706.06083)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
