#!/usr/bin/env python3
"""
Find tokens whose byte encodings can generate a target UTF-8 string
via prefix overlap or containment, using the tokenizer of a specified
language model.
"""
from transformers import AutoTokenizer


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unficode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.

    copy of
    https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9C1-L28C29
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def encode_str_utf8(any_string):
    if type(any_string) == str:
        return any_string.encode("utf-8")
    elif type(any_string) == bytes:
        return any_string
    else:
        raise TypeError()


class TokenByteFinder:
    def __init__(self, tokenizer):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

        unicode_to_bytes = {ord(v): k for k, v in bytes_to_unicode().items()}
        self.token_bytes = {
            t_id: bytes([unicode_to_bytes[ord(t_char)] for t_char in t])
            for t, t_id in self.vocab.items()
        }

    def get_singe_token(self, char):
        single_id = self.tokenizer.encode(char, add_special_tokens=False)
        if len(single_id) == 1:
            return single_id[0]
        raise AssertionError("Char encoded is not a single token")




    def find_prefix_overlaps(self, target_bytes):
        target_bytes = encode_str_utf8(target_bytes)

        prefix_overlaps = {}
        for tok_id, b in self.token_bytes.items():
            max_k = min(len(b), len(target_bytes) - 1)
            for k in range(1, max_k + 1):
                if b[-k:] == target_bytes[:k]:
                    prefix_overlaps[tok_id] = k
                    break
        return prefix_overlaps

    def find_containment_tokens(self, target_bytes):
        target_bytes = encode_str_utf8(target_bytes)

        containment = {
            tok_id: True for tok_id, b in self.token_bytes.items()
            if target_bytes in b
        }
        return containment

    def get_generating_tokens(self, any_string):
        target_bytes = encode_str_utf8(any_string)
        prefix_overlaps = self.find_prefix_overlaps(target_bytes)
        containment = self.find_containment_tokens(target_bytes)
        
        # Merging keys from both dictionaries and removing duplicates
        generating_tokens = list(set(prefix_overlaps.keys()) | set(containment.keys()))
        return generating_tokens

    def return_token_bytes(self, ids):
        if type(ids) == int:
            return self.token_bytes[ids]

        if type(ids) == list:
            return [self.token_bytes[i] for i in ids]

        return None


def bytes_to_hex(b: bytes) -> str:
    return " ".join(f"0x{byte:02x}" for byte in b)

def safe_utf8(b: bytes) -> str:
    """Best-effort UTF-8 decoding (shows  for invalid sequences)."""
    return b.decode("utf-8", errors="replace")

def print_token_info(tok_id, oracle, k=None):
    b = oracle.token_bytes[tok_id]
    tok_str = oracle.id_to_token[tok_id]
    print(f"Token ID     : {tok_id}")
    print(f"Token string : {repr(tok_str)}")
    print(f"Token UTF    : {repr(safe_utf8(b))}")
    print(f"Token bytes  : {bytes_to_hex(b)}")
    if k is not None:
        print(f"Overlap size : {k} byte(s)")
        print(f"Overlap bytes: {bytes_to_hex(b[-k:])}")

def print_results(oracle, TARGET_UTF):
    target_bytes = TARGET_UTF.encode("utf-8")
    prefix_overlaps = oracle.find_prefix_overlaps(target_bytes)
    containment = oracle.find_containment_tokens(target_bytes)

    print("=" * 80)
    print("MODEL")
    print("=" * 80)
    # The model name is not stored in the oracle, so it can't be printed here.
    # If needed, it should be passed as an argument.
    # print(oracle.model_name)
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
        for tok_id, k in sorted(prefix_overlaps.items(), key=lambda x: -x[1]):
            print_token_info(tok_id, oracle, k)
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
            print_token_info(tok_id, oracle)
            print("-" * 60)

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Examples:
    #   "meta-llama/Llama-2-7b-hf"
    #   "mistralai/Mistral-7B-v0.1"
    #   "gpt2"
    #   "EleutherAI/gpt-neox-20b"

    TARGET_UTF = "€" # "—"  "‡" try: "€", "—", "🙂", "\u0301"

    oracle = TokenByteFinder(MODEL_NAME)
    
    # The get_generating_tokens method can be used to get the combined list of tokens
    generating_tokens = oracle.get_generating_tokens(TARGET_UTF)
    # print(f"Generating Tokens: {generating_tokens}")

    # The printing of results is done by calling print_results
    print_results(oracle, TARGET_UTF)
