#!/usr/bin/env python3
"""
Find tokens whose byte encodings can generate a target UTF-8 string
via prefix overlap or containment, using the tokenizer of a specified
language model.
"""
from transformers import AutoTokenizer

# ============================================================
# USER INPUT
# ============================================================

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# Examples:
#   "meta-llama/Llama-2-7b-hf"
#   "mistralai/Mistral-7B-v0.1"
#   "gpt2"
#   "EleutherAI/gpt-neox-20b"

TARGET_UTF =  "—" # "‡"  # try: "€", "—", "🙂", "\u0301"


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unficode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

unicode_to_bytes_dict = {ord(v): k for k, v in bytes_to_unicode().items()}
def unicode_to_bytes(target):
    return unicode_to_bytes_dict[target]


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)

vocab = tokenizer.get_vocab()
id_to_token = {idx: tok for tok, idx in vocab.items()}

token_bytes = {
    t_id: bytes([unicode_to_bytes(ord(t_char)) for t_char in t])
    for t, t_id in vocab.items()
}

target_bytes = TARGET_UTF.encode("utf-8")

space_id = tokenizer(" ", add_special_tokens=False)["input_ids"]
special_id = tokenizer("‡", add_special_tokens=False)["input_ids"]
both_id = tokenizer(" ‡", add_special_tokens=False)["input_ids"]
space_id, special_id, both_id


def show_bytes_list(label, ids, tokenizer):
    print(f"Label {label}")
    ids = [ids] + ids

    for id_i in ids:
        tok_str = tokenizer.convert_ids_to_tokens(id_i)
        if type(tok_str) != list:
            tok_str = [tok_str]

        for t in tok_str:
            b = t.encode("utf-8", errors="surrogatepass")
            print(f"id={id_i}, bytes={[hex(x) for x in b]}, tok_str={t}")

def show_bytes(label, ids):
    bytes = [token_bytes[id_i] for id_i in ids]
    print(f"Label {label}, ids={ids}, bytes={bytes}")

print(f"Target bytes: {target_bytes}")

show_bytes("space", space_id)
show_bytes("special", special_id)
show_bytes("both", both_id)


def bytes_to_hex(b: bytes) -> str:
    return " ".join(f"0x{byte:02x}" for byte in b)


def safe_utf8(b: bytes) -> str:
    """Best-effort UTF-8 decoding (shows � for invalid sequences)."""
    return b.decode("utf-8", errors="replace")


# ============================================================
# FIND PREFIX-OVERLAP TOKENS
# ============================================================

prefix_overlaps = []

for tok_id, b in token_bytes.items():
    max_k = min(len(b), len(target_bytes) - 1)
    for k in range(1, max_k + 1):
        if b[-k:] == target_bytes[:k]:
            prefix_overlaps.append((tok_id, k))
            break

# ============================================================
# FIND CONTAINMENT TOKENS
# ============================================================

containment = [
    tok_id for tok_id, b in token_bytes.items()
    if target_bytes in b
]

# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("MODEL")
print("=" * 80)
print(MODEL_NAME)
print()

print("=" * 80)
print("TARGET")
print("=" * 80)
print(f"UTF string : {repr(TARGET_UTF)}")
print(f"UTF-8 bytes: {bytes_to_hex(target_bytes)}")
print()

# ---------- Prefix overlaps ----------
print("=" * 80)
print("PREFIX OVERLAPS (token ends with prefix of target)")
print("=" * 80)

if not prefix_overlaps:
    print("None found.")
else:
    for tok_id, k in sorted(prefix_overlaps, key=lambda x: -x[1]):
        b = token_bytes[tok_id]
        tok_str = id_to_token[tok_id]
        print(f"Token ID     : {tok_id}")
        print(f"Token string : {repr(tok_str)}")
        print(f"Token UTF    : {repr(safe_utf8(b))}")
        print(f"Token bytes  : {bytes_to_hex(b)}")
        print(f"Overlap size : {k} byte(s)")
        print(f"Overlap bytes: {bytes_to_hex(b[-k:])}")
        print("-" * 60)

# ---------- Containment ----------
print()
print("=" * 80)
print("CONTAINMENT (token contains full target bytes)")
print("=" * 80)

if not containment:
    print("None found.")
else:
    for tok_id in containment:
        b = token_bytes[tok_id]
        tok_str = id_to_token[tok_id]
        print(f"Token ID    : {tok_id}")
        print(f"Token string: {repr(tok_str)}")
        print(f"Token UTF   : {repr(safe_utf8(b))}")
        print(f"Token bytes : {bytes_to_hex(b)}")
        print("-" * 60)
