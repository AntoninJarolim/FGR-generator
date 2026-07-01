from transformers import AutoTokenizer
import unicodedata

from huggingface_hub import HfApi

api = HfApi()

user = api.whoami()
print("You are logged in as:", user["name"])


# MODEL_NAME = "google/gemma-3-4b-it"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CANDIDATES = [
    "§", "‡", "¶", "※", "⁂", "¤", "↯",
    "⟦", "⟧", "⟨", "⟩", "⟪", "⟫",
    "〚", "〛", "⌈", "⌉", "⌊", "⌋"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"\nModel: {MODEL_NAME}")
print("Tokenizer class:", tokenizer.__class__.__name__)
print("-" * 50)

def is_single_token(char):
    ids = tokenizer.encode(char, add_special_tokens=False)
    return len(ids) == 1, ids

single_tokens = []

for c in CANDIDATES:
    ok, ids = is_single_token(c)
    if ok:
        single_tokens.append((c, ids[0]))
        print(f"[OK] '{c}'  → token_id: {ids[0]}  ({unicodedata.name(c)})")
    else:
        print(f"[NO] '{c}'  → split into {ids}")

# ----------------------------
# FAIL IF NOT FOUND
# ----------------------------
print(single_tokens)
