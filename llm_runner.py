import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random


def print_first_logits(all_logits, first_logits):
    print(torch.allclose(
        first_logits,
        all_logits,
        atol=1e-3
    ))

    print(torch.allclose(
        first_logits,
        all_logits,
        atol=1e-5
    ))

    a = first_logits
    b = all_logits

    mask = ~torch.isclose(a, b, atol=1e-6)

    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    print("num differing:", idx.numel())
    print("first 10 indices:", idx[:10])

    for i in idx[:10]:
        print(
            i.item(),
            a[i].item(),
            b[i].item(),
            (a[i] - b[i]).item()
        )


class LLMRunner:
    def __init__(self, model_name, device=None, max_new_tokens=512, do_sample=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )

        self.model.eval()
        torch.set_grad_enabled(False)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        self._setup_model_config()

    def _setup_model_config(self):
        # Disable processing logits in any wa
        self.model.generation_config.top_k = 0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.do_sample = False

        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tokenize_char(self, character):
        assert len(character) == 1

        tokenized = self.tokenizer(character, return_tensors="pt", add_special_tokens=False)
        assert len(tokenized["input_ids"][0]) == 1, "This character is not tokenized as one token."

        return tokenized["input_ids"][0]

    def tokenize_run(self, prompt: str) -> str:
        """
        Tokenizes the prompt, runs the model autoregressively, and returns decoded text.
        """
        # Encode the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate output
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            bad_words_ids=None,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )

        logits = torch.stack(generated.scores, dim=1)

        # first_logits = generated.scores[0][0]
        # all_logits = self.model(**inputs).logits[0, -1, :]
        # self.print_first_logits(all_logits, first_logits)

        # Decode to string
        return logits, self.decode_generated(inputs, generated.sequences)

    def decode_generated(self, inputs, generated):
        input_size = len(inputs['input_ids'][0])
        generated_text = self.tokenizer.decode(generated[0][input_size:], skip_special_tokens=True)
        return generated_text

    def encode_ctx(self, template, context) -> str:
        # tokenize template and context independently
        tmpl_enc = self.tokenizer(
            template,
            return_tensors="pt"
        )

        if context[0] != " ":
            context = " " + context

        ctx_enc = self.tokenizer(
            context,
            add_special_tokens=False,
            return_tensors="pt"
        )

        context_len = ctx_enc["input_ids"].size(1)

        input_ids = torch.cat(
            [tmpl_enc["input_ids"], ctx_enc["input_ids"]],
            dim=1
        ).to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            # attention_mask=attention_mask
        )

        return outputs['logits'][:, -context_len - 1:]

    def append_eos_token_id(self, tokens):
        eos_id = self.tokenizer.eos_token_id
        eos = torch.tensor([[eos_id]], device=tokens.device)
        ret = torch.cat([tokens, eos], dim=1)
        return ret

    def split_context_by_token_pos(self, text: str, start_context: int):
        """
        Returns text split by position of token given by start_context parameter
        """
        # Tokenize without special tokens
        ctx = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        )

        # Get token ids (1D tensor)
        token_ids = ctx["input_ids"][0]

        # Split tokens
        left_tokens = token_ids[:start_context]
        right_tokens = token_ids[start_context:]

        # Decode back to text
        left_text = self.tokenizer.decode(left_tokens, skip_special_tokens=True)
        right_text = self.tokenizer.decode(right_tokens, skip_special_tokens=True)

        return left_text, right_text
