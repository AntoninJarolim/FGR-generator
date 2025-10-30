import json
import os
from json import JSONDecodeError

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import text_utils


class ExplanationsDataset(Dataset):

    def __init__(self, file_path, tokenizer,
                 decode_positive_as_list=False, error_on_invalid=False,
                 psg_key='psg_text', spans_key='selected_spans'):
        self.psg_key = psg_key
        self.spans_key = spans_key
        self.invalid_indexes = []
        self.error_on_invalid = error_on_invalid
        self.decode_positive_as_list = decode_positive_as_list
        self.tokenizer = tokenizer
        self.file_path = file_path
        # Load the entire JSONL file into memory
        with open(file_path, 'r', encoding='utf-8') as f:
            def decode_line(line):
                try:
                    return json.loads(line)
                except JSONDecodeError as e:
                    print(e)
                    print(line)

            self.data = [decode_line(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Each sample has:
        'psg_id' passage and 'selected_spans', which needs to be converted to binary vector
        of relevance indications
        :param idx:
        :return:
        """
        sample: dict = self.data[idx]
        try:
            tokenized_positive, binary_rationales = self.tokenize_with_spans(
                sample[self.psg_key], sample[self.spans_key]
            )

        except AssertionError as e:
            if self.error_on_invalid:
                raise e
            self.invalid_indexes.append(idx)
            print(f"Error in finding spans: {e}")
            return None, None

        return {
            self.psg_key: sample[self.psg_key],
            'tokenized_positive': tokenized_positive,
            'tokenized_positive_decoded': self.decode_list(tokenized_positive),
            'rationales': binary_rationales,
            'selected_spans': sample[self.spans_key]
        }

    def encode_text(self, text):
        outputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        outputs.update({
            "offset_mapping": outputs["offset_mapping"].squeeze(0),
            "input_ids": outputs["input_ids"].squeeze(0),
        })
        return outputs

    def tokenize_with_spans(self, positive_text, selected_spans):
        # Find spans and extract starts and ends
        found_spans = text_utils.find_spans(positive_text, selected_spans)

        spand_starts = torch.tensor([s['start'] for s in found_spans])
        spand_ends = torch.tensor([s['end'] for s in found_spans])

        # Encode and split offsets to starts and ends
        encoded = self.encode_text(positive_text)
        encoded_text = encoded["input_ids"]
        offset_mapping_starts = encoded["offset_mapping"][:, 0]  # 0th dim are starts
        offset_mapping_ends = encoded["offset_mapping"][:, 1]  # 1st dim are ends

        # Reshape for broadcasting each-to-each
        start_overlap = offset_mapping_starts[:, None] < spand_ends
        end_overlap = offset_mapping_ends[:, None] > spand_starts
        label_tensors = torch.any(start_overlap & end_overlap, dim=1)

        # Binary labels must match encoded text length in number of tokens
        assert len(encoded_text) == len(label_tensors)
        return encoded_text, label_tensors

    def decode_list(self, tokens):
        return (
            text_utils.tokenize_list(tokens, self.tokenizer)
            if self.decode_positive_as_list
            else None
        )


if __name__ == "__main__":
    # Example usage
    # file_path = 'data/35_random_samples_explained.jsonl'
    # file_path = 'data/35_random_samples_Meta-Llama-3.1-8B-Instruct.jsonl'
    # file_path = 'data/35_random_samples_gpt-4o-mini-2024-07-18.jsonl'
    # file_path = 'data/35_random_samples_gpt-4o-2024-08-06.jsonl'
    # file_path = 'data/35_random_samples_llama3.1:70b-instruct-q4_0.jsonl'
    # file_path = 'data/35_random_samples_llama2:13b.jsonl'
    # file_path = 'data/gemma2:27b-instruct-q8_0.jsonl'
    # file_path = 'data/triplets_explained.jsonl'

    dataset_name  = 'gemma3:4b_from0-to185'
    file_path = f'data/extracted_relevancy/{dataset_name}.jsonl'

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    dataset = ExplanationsDataset(
        file_path, tokenizer, decode_positive_as_list=True,
        psg_key='clean_text', error_on_invalid=True
    )

    # Iterate through the data
    exceptions = []
    for i in tqdm(range(len(dataset))):
        try:
            d = dataset[i]
        except Exception as e:
            exceptions.append((i, e))

    exceptions_dir = f"data/failed_explanations/{dataset_name}"
    os.makedirs(exceptions_dir, exist_ok=True)

    # Save exceptions to files
    for idx, exception in exceptions:
        with open(f"{exceptions_dir}/exception_{idx}.txt", "w") as f:
            f.write(str(exception))

