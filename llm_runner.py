import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.plotting_logits import plot_series


class LLMRunner:
    def __init__(self, model_name, device=None, max_new_tokens=1024, do_sample=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )

        self.model.eval()
        torch.set_grad_enabled(False)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

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
            do_sample=self.do_sample
        )

        # Decode to string
        return self.decode_generated(inputs, generated)

    def tokenize_run_custom(self, prompt: str, targets_text=None, special=None, teacher_forcing=False):
        """
        Tokenizes the prompt, runs the model autoregressively, and returns decoded text.
        But without using model.generate
        """
        # Encode the input
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        targets = self.tokenizer(targets_text,
                                 return_tensors="pt",
                                 add_special_tokens=False,
                                 ).to(self.device)["input_ids"]

        special_id = self.tokenize_char(special)

        input_ids = inputs["input_ids"]
        generated = input_ids

        nr_new_tokens = len(targets[0]) if teacher_forcing else self.max_new_tokens

        special_logits_list = []
        for new_pos in range(nr_new_tokens + 1):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated, # [:, -1:] if past_key_values is not None else generated,
                    # attention_mask=attention_mask,
                    # past_key_values=past_key_values,
                    # use_cache=False,
                )

            # To not break following indexing
            if new_pos == nr_new_tokens:
                break

            # (batch, sq_len, vocab)
            next_token = (
                targets[:, new_pos]
                if teacher_forcing else
                torch.argmax(outputs.logits, dim=-1)[:, -1]
            )

            if next_token.item() == special_id.item():
                pass
            else:
                special_id_logit = outputs.logits[:, -1, special_id]
                special_logits_list.append(special_id_logit.item())

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            # optional stopping on EOS
            if self.tokenizer.eos_token_id is not None:
                if (next_token == self.tokenizer.eos_token_id).all():
                    special_logits_list.pop()
                    break

        # todo: encode with one self.model() call
        all_logits = self.encode_ctx(prompt, targets_text)
        selected_logits = all_logits[0, :, special_id].tolist()
        # Log the logits for a special token

        max_tokens = 50
        plot_series(
            [
                (selected_logits[:max_tokens], "At once"),
                (special_logits_list[:max_tokens], "one by one"),
                (outputs.logits[0, -len(targets[0])-1:, special_id].tolist()[:max_tokens], "one by one at the end"),
            ]
        )

        return self.decode_generated(inputs, generated)

    def decode_generated(self, inputs, generated):
        input_size = len(inputs['input_ids'][0])
        generated_text = self.tokenizer.decode(generated[0][input_size:], skip_special_tokens=True)
        return generated_text

    def encode_ctx(self, template, context) -> str:
        # tokenize template and context independently
        tmpl_enc = self.tokenizer(
            template,
            add_special_tokens=False,
            return_tensors="pt"
        )

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

        return outputs['logits'][:, -context_len-1:]

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
