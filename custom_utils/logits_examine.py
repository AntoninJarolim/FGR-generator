
import torch


def pretty_token(tok_id, tok_val, tok_bytes, tokenizer, max_bytes=20):
    decoded = tokenizer.decode([tok_id])
    b = tok_bytes[:max_bytes]
    b_str = b.hex(" ") + (" …" if len(tok_bytes) > max_bytes else "")
    return f"id={tok_id:>6}  logit={tok_val:>7.3f}  tok={decoded!r:<12}  bytes=[{b_str}]"


def print_topk_logits_decoded(tokens_finder, tokenizer, logits, start_span_tokens_id=None, top_k_tokens = 10):

    for time in range(logits.shape[1]):
        topk_vals, topk_ids = torch.topk(logits[0, time], top_k_tokens)
        topk_bytes = tokens_finder.return_token_bytes(topk_ids.tolist())

        print(f"\nTime {time} {start_span_tokens_id}")
        for v, i, b in zip(topk_vals.tolist(), topk_ids.tolist(), topk_bytes):
            print(" ", pretty_token(i, v, b, tokenizer))