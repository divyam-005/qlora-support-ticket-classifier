# QLoRA Support Ticket Classifier

Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA for structured customer-support ticket classification, then merged, converted to GGUF, quantized to Q4_K_M, and served locally with Ollama.

## Pipeline

```text
Qwen2.5-1.5B-Instruct
→ Baseline Evaluation
→ 4-bit NF4 + LoRA
→ QLoRA Fine-Tuning
→ Evaluation
→ Merge Adapter
→ GGUF / Q4_K_M
→ Ollama
→ Python HTTP Client
```

## Results

| Metric | Base | QLoRA |
|---|---:|---:|
| JSON validity | 100% | 100% |
| Category accuracy | 70% | 100% |
| Priority accuracy | 50% | 80% |

Results are from 10 held-out examples and are demonstration-level, not a large benchmark.

## Run

```bash
python src/generate_dataset.py
python src/baseline.py
python src/train_qlora.py
python src/evaluate.py
python src/merge_adapter.py
```

For local serving:

```bash
ollama create qwen-support -f ollama/Modelfile
ollama run qwen-support
python src/ollama_client.py
```

## Tech

`Python` `PyTorch` `Transformers` `PEFT` `TRL` `bitsandbytes` `Qwen2.5` `QLoRA` `GGUF` `Ollama`
