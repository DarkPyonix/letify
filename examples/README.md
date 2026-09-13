# Examples

Working scenarios, in the order a real project meets them.

| Scenario | Shows |
|---|---|
| [nvfp4-lora/](nvfp4-lora/) ([Korean](nvfp4-lora/README_ko.md)) | Fine-tuning a model with LoRA on a rented Blackwell GPU: one smoke call, a hyperparameter sweep across pooled sessions, checkpoints into a content addressed volume, and resuming after a preemption |

Every script names the provider by alias and nothing else, so the same file runs against a
Colab runtime, a lab machine over SSH or this laptop by changing one line of `.letify`.
Start with the local provider, which needs no account:

```bash
cd nvfp4-lora
python 00_smoke.py --provider local
```
