import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_FILE = "data/test_data.json"

SYSTEM = """You are a customer support ticket classifier.
Allowed categories: payment_issue, refund_request, delivery_issue, account_access, technical_issue, billing_issue, other.
Allowed priorities: low, medium, high.
Return ONLY valid JSON with exactly these keys: category, priority, summary."""

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    with open(TEST_FILE, encoding="utf-8") as f:
        tests = json.load(f)

    valid = cat = pri = 0
    for x in tests:
        prompt = tok.apply_chat_template(
            [{"role":"system","content":SYSTEM},{"role":"user","content":x["ticket"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            pred = json.loads(raw)
            valid += 1
            cat += pred.get("category") == x["expected"]["category"]
            pri += pred.get("priority") == x["expected"]["priority"]
        except json.JSONDecodeError:
            pass

    n = len(tests)
    print(f"JSON validity: {valid/n:.0%}")
    print(f"Category accuracy: {cat/n:.0%}")
    print(f"Priority accuracy: {pri/n:.0%}")

if __name__ == "__main__":
    main()
