import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from utils.plotting_logits import plot_series
import random

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

    def print_first_logits(self, all_logits, first_logits):
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
                (outputs.logits[0, -len(targets[0]) - 1:, special_id].tolist()[:max_tokens], "one by one at the end"),
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

    def encode_ctx_diff(self, template, context) -> str:
        # tokenize template and context independently
        tmpl_enc = self.tokenizer(
            template,
            add_special_tokens=False,
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
        outputs = self.encode_ctx(template, context)

        # add eos and shift to right by one
        # Context has size X
        context_ids = input_ids[:, -context_len:]

        # GT_ids has size X+1, cuz we add EOS to the right
        GT_ids = self.append_eos_token_id(context_ids)

        # output_logits has size X+1, cuz we add logit from ctx_len-1 token
        output_logits = outputs['logits'][:, -context_len - 1:]

        # Unsqueeze to match size of output_logits
        gt_logits = output_logits.gather(dim=-1, index=GT_ids.unsqueeze(-1))

        predicted_ids = torch.argmax(output_logits, dim=-1)

        return output_logits - gt_logits, predicted_ids

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
