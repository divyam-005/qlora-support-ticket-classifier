import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/qlora_support_classifier_adapter"
MERGED_MODEL_PATH = "/content/drive/MyDrive/qlora_support_classifier_merged"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    print(f"Merged model saved to: {MERGED_MODEL_PATH}")

if __name__ == "__main__":
    main()
