import json
import logging
import re
import socket

import numpy as np
import torch

logger = logging.getLogger('main')


def conv_to_torch(pred, labels):
    if type(pred) is not torch.Tensor:
        pred = torch.tensor(pred)
    if type(labels) is not torch.Tensor:
        labels = torch.tensor(labels)
    return pred, labels


def mrr_metric(preds, labels):
    """
    Compute Mean Reciprocal Rank (MRR).

    Arguments:
    pred -- Tensor of predicted scores.
    labels -- Tensor of true labels.

    Returns:
    MRR -- Mean Reciprocal Rank for the batch.
    """

    # skip computation if all labels are 0
    if sum(labels) == 0:
        return 0

    preds, labels = conv_to_torch(preds, labels)
    sorted_indices = torch.argsort(preds, descending=True)
    sorted_labels = labels[sorted_indices]
    rank = np.argwhere(sorted_labels.cpu()).flatten().item() + 1
    # Calculate the reciprocal rank
    return 1.0 / rank


def recall_metric(sorted_labels, ks):
    if type(ks) is not list:
        if type(ks) is int:
            ks = [ks]
        else:
            raise AssertionError("Type of ks parameter has to be list or int.")

    recall_ks = [sum(sorted_labels[:k]) for k in ks]
    return recall_ks


def pred_recall_metric(pred, labels, ks):
    pred, labels = conv_to_torch(pred, labels)

    # skip computation if all labels are 0
    if sum(labels) == 0:
        return torch.zeros(len(ks))

    sorted_indexes = torch.argsort(pred, descending=True)
    sorted_labels = labels[sorted_indexes]
    return recall_metric(sorted_labels, ks)


def compute_recall(path, ks):
    with open(path, 'r') as file:
        recalls = []
        for line in file:
            dpr_pairs = json.loads(line.strip())
            sorted_labels = [int(x['label']) for x in dpr_pairs]
            recall_ks = recall_metric(sorted_labels, ks)
            recalls.append(recall_ks)

        recalls = torch.tensor(recalls)
        N = recalls.shape[0]
        sum_recalls = torch.sum(recalls, dim=0) / N
    return sum_recalls


def transform_batch(batch, take_n=0, device=None):
    if take_n > 0:
        batch = batch[:take_n]

    # Transform each item and make it tensor if not already
    def get_value_tensor(source_item, key):
        value_out = source_item[key]
        if not isinstance(value_out, list):
            value_out = [value_out]
        if not isinstance(value_out, torch.Tensor):
            value_out = torch.tensor(value_out, device=device)
        return value_out

    # Gather 'label', 'in_ids', and 'att_mask' from these members
    labels = torch.vstack([get_value_tensor(item, 'label') for item in batch]).flatten()
    in_ids = torch.vstack([get_value_tensor(item, 'in_ids') for item in batch])
    att_masks = torch.vstack([get_value_tensor(item, 'att_mask') for item in batch])
    tt_ids = torch.vstack([get_value_tensor(item, 'tt_ids') for item in batch])

    # Combine into a dictionary as expected by your code
    return {
        'label': labels,
        'in_ids': in_ids,
        'att_mask': att_masks,
        'tt_ids': tt_ids
    }


def calc_physical_batch_size(batch_size, gpu_batches=None):
    """
    Calculates logical and physical batch size based on the GPU configuration.
    :param batch_size: Actual batch size
    :param gpu_batches: Json dict with GPU name as key and physical batch size as value
    :return: Tuple of logical and physical batch size
    """
    assert batch_size > 0, "Batch size should greater than 0"

    default_gpu_batches = {
        "NVIDIA GeForce RTX 2080 Ti": 1,
        "NVIDIA GeForce RTX 3090": 4,
        "NVIDIA RTX A5000": 4,
    }
    gpu_batches = gpu_batches or default_gpu_batches

    # Get the host name
    host_name = socket.gethostname()
    hosts_gpu_file = "../.host_config.json"  # Json dict with host name as key and gpu name as value
    gpu_hosts = json.load(open(hosts_gpu_file))
    assert host_name in gpu_hosts, f"Host {host_name} not found in {hosts_gpu_file} file"

    # Calculate logical and physical batch size
    train_batch_size = gpu_batches[gpu_hosts[host_name]]
    gradient_accumulation_steps = batch_size // train_batch_size

    # Do not accumulate gradients if the batch size is less than the physical batch size
    if train_batch_size >= batch_size:
        return batch_size, 1

    return train_batch_size, gradient_accumulation_steps


def load_model(cross_encoder, load_path):
    cross_encoder.load_state_dict(torch.load(load_path))
    logger.info(f"Model loaded successfully from {load_path}")
    return cross_encoder


def save_model(cross_encoder, save_model_path, msg_str=None):
    torch.save(cross_encoder.state_dict(), save_model_path)
    msg_str = msg_str if msg_str is not None else "Model saved to "
    logger.info(f"{msg_str} {save_model_path}")


def save_best_model(cross_encoder, save_model_path):
    save_model(cross_encoder, save_model_path, "New best saved to ")


class LimitedDataLoader:
    def __init__(self, dataloader, max_iterations):
        self.dataloader = dataloader
        self.max_iterations = max_iterations

    def __iter__(self):
        self.iter_loader = iter(self.dataloader)
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.max_iterations:
            self.counter += 1
            return next(self.iter_loader)
        else:
            raise StopIteration

    def __len__(self):
        return min(self.max_iterations, len(self.dataloader))


def find_reference_span(passage, grounding_references):
    try:
        before, after = passage.split(grounding_references, 1)
    except ValueError:
        passage = re.sub(' +', ' ', passage)
        passage = re.sub(' ,', ',', passage)
        before, after = passage.split(grounding_references, 1)
    return before, after


def create_grounding_annt_list(passage, grounding_references, label, tokenizer, return_failed=False):
    if label == 0:
        return split_to_tokens(passage, tokenizer), None

    # Create annotation list
    failed_references = []
    annotation_list = []
    broken_passage = []
    for reference in grounding_references:
        # strip space, because it is sometimes appended at the end and the space is not in
        # the passage, leading to not finding part of passage containing this reference span
        ref_span = reference["ref_span"].strip(" ")

        if passage == ref_span:
            annotation_list.append(True)
            broken_passage.append(ref_span)
            break

        try:
            before, after = find_reference_span(passage, ref_span)
        except ValueError:
            failed_references.append(ref_span)
            continue

        # Found reference is not at the beginning of the passage
        if before != "":
            annotation_list.append(False)
            broken_passage.append(before)  # Do not label this part of passage as GT

        annotation_list.append(True)
        broken_passage.append(ref_span)

        passage = after
        if passage == "":
            break

    # Append the remaining part of passage (if any)
    if passage != "":
        annotation_list.append(False)
        broken_passage.append(passage)

    unpack_passages = []
    unpack_annt_list = []
    for psg, annt in zip(broken_passage, annotation_list):
        psg_tokens = split_to_tokens(psg, tokenizer)
        for t in psg_tokens:
            unpack_passages.append(t)
            unpack_annt_list.append(annt)

    # Do not return empty list but None, pass it to the function as None
    if not unpack_passages:
        unpack_passages = None

    if return_failed:
        return unpack_passages, unpack_annt_list, failed_references
    return unpack_passages, unpack_annt_list


def split_to_tokens(text, tokenizer):
    return tokenizer.batch_decode(tokenizer(text)["input_ids"])[1:][:-1]


def create_highlighted_word(word, bg_color, fg_color):
    # Create the HTML string with inline CSS for styling
    return f"""
        <span style="display: 
        inline-flex; 
        flex-direction: row; 
        align-items: center; 
        background: {bg_color}; 
        border-radius: 0.5rem; 
        padding: 0.07rem 0.15rem; 
        overflow: hidden; 
        line-height: 1; 
        color: {fg_color};
        ">                
        {word}
        </span>
    """


def create_highlighted_passage(passage_tokens, gt_label_list, annotation_scores,
                               base_colour, colour_type, gt_label_colour=None):
    highlighted_passage = []
    gt_label_colour = gt_label_colour if gt_label_colour is not None else "#00A50F"

    # Create default list of colours for each token
    colours_annotation_list = ["#00000000"] * len(passage_tokens)
    if annotation_scores is not None:
        colours_annotation_list = annt_list_2_colours(annotation_scores, base_colour, colour_type)

    if gt_label_list is None:
        gt_label_list = [None] * len(passage_tokens)

    for bg_colour, token, gt_label in zip(colours_annotation_list, passage_tokens, gt_label_list):
        fg_colour = "#000000" if not gt_label else gt_label_colour
        token = token.replace('$', '\$')
        span_text = create_highlighted_word(token, bg_colour, fg_colour)
        highlighted_passage.append(span_text)

    return highlighted_passage


def annt_list_2_colours(annotation_list, base_colour, colours):
    if annotation_list is None:
        return None

    if not isinstance(annotation_list, torch.Tensor):
        annotation_list = torch.Tensor(annotation_list)

    eps = 1e-6
    normalized_tensor_list = annotation_list / (torch.max(annotation_list) + eps)
    if colours == "nonlinear":
        negative_index = torch.where(normalized_tensor_list < 0)
        normalized_tensor_list = torch.abs(normalized_tensor_list)
        transf_list = -torch.log(normalized_tensor_list)
        normalized_tensor_list = 1 - (transf_list / torch.max(transf_list))
        normalized_tensor_list[negative_index] = -normalized_tensor_list[negative_index]

    colour_range = (normalized_tensor_list * 255).type(torch.int64)

    if not (-256 <= torch.min(colour_range)):
        f"min: Conversion of {torch.min(colour_range)} to colour range failed"
    assert torch.max(colour_range) < 256, f"max: Conversion of {torch.max(colour_range)} to colour range failed"

    if base_colour == "blue":
        def conv_fce(x):
            return f'#1111{x:02x}'
    elif base_colour == "green":
        def conv_fce(x):
            if x > 0:
                return f'#77{x:02x}77'
            else:
                return f'#{abs(x):02x}1111'
    elif base_colour == "red":
        def conv_fce(x):
            return f'#{x:02x}1111'
    else:
        raise ValueError(f"Base colour {base_colour} not supported")

    coloured_list = [conv_fce(x) for x in colour_range]
    return [x if x not in ["#000000", "#111100", "#11011"] else ["#00000000"]
            for x in coloured_list]