import argparse
import json
import os
import time
import re
from datetime import datetime

from jinja2 import Template
from jsonlines import jsonlines
from openai import OpenAI, APITimeoutError
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from explainable_dataset import ExplanationsDataset
from custom_utils.longembed import is_longembed, load_longembed



class OpenAIGenerator:
    # Seconds to wait between retries of a timed-out request, so a busy server
    # is not hammered the moment it refused us.
    RETRY_SLEEP_S = 30

    def __init__(self, model_name, generation_client=False, api_token_env_var=None,
                 max_retries=50):
        url = os.getenv('OPENAI_BASE_URL') + '/v1'
        if generation_client == 'ollama':
            print("Initialized OLLAMA generation client.")
            self.client = OpenAI(
                base_url=url,
                api_key=os.getenv(api_token_env_var, 'ollama'),
            )
        else:
            self.client = OpenAI()
        self.model = model_name  # self.model = "gpt-4o-2024-08-06" "gpt-4o-mini"
        self.max_retries = max_retries

        self.temperature = 0.2
        self.max_tokens = 2**14 # 16384
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.system_message = "You are helpful linguistic specialist eager to complete given task."

    def create_message(self, system_prompt, user_prompt):
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    def create_api_call_dict(self, system_prompt, user_prompt):
        """
        Create API call for OpenAI API from a rendered (system, user) prompt pair.
        """
        return dict(
            model=self.model,
            messages=self.create_message(system_prompt, user_prompt),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            response_format={
                "type": "json_object"
            }
        )

    def __call__(self, system_prompt, user_prompt):
        api_dict = self.create_api_call_dict(system_prompt, user_prompt)
        # A busy server (e.g. e-INFRA under load, where our requests may have
        # low priority) surfaces as APITimeoutError. Retry instead of crashing
        # the whole run -- a quiet window (e.g. at night) will let it through.
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**api_dict)
            except APITimeoutError:
                if attempt == self.max_retries:
                    tqdm.write(f"Request timed out and all {self.max_retries} retries "
                               f"are exhausted, giving up.")
                    raise
                tqdm.write(f"Request timed out, retrying in {self.RETRY_SLEEP_S}s "
                           f"(retry {attempt + 1}/{self.max_retries}).")
                time.sleep(self.RETRY_SLEEP_S)


# JSON schema for the {"spans": [...]} contract that the downstream parser
# (custom_utils/text_utils.find_spans / decode_one) expects: a list of verbatim
# substring strings. Used for optional vLLM guided (constrained) decoding.
SPANS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["spans"],
}


class VLLMGenerator:
    """Native vLLM offline batched inference (LLM / SamplingParams).

    Unlike OpenAIGenerator this needs no server: the model weights are loaded
    in-process and a whole batch of prompts is generated in a single
    ``llm.generate`` call. Its ``generate`` method returns the list of raw
    message-content strings aligned to the input prompts, which
    ``generate_one_batch_vllm`` then writes in the same output-file schema the
    rest of the pipeline consumes.
    """

    # Tokens held back below max_model_len when fitting a prompt, to absorb
    # chat-template special tokens and tokenizer boundary effects
    TRUNC_MARGIN = 256

    def __init__(self, model_name, max_model_len=None, gpu_memory_utilization=0.9,
                 tensor_parallel_size=None, dtype='auto', max_gen_tokens=2**16,
                 guided_json=False, psg_key='passage'):

        # vLLM does not shard across GPUs on its own -- default to every visible
        # GPU so the run uses two (or more) cards when they are available.
        if tensor_parallel_size is None:
            import torch
            tensor_parallel_size = max(1, torch.cuda.device_count())

        print(f"Initialized native vLLM offline generation client "
              f"(tensor_parallel_size={tensor_parallel_size}).")
        self.model = model_name
        self.max_gen_tokens = max_gen_tokens
        self.psg_key = psg_key
        self._n_truncated = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
        # Use the context length the engine actually settled on (it may cap the
        # requested value to what fits in the KV cache), so passage truncation
        # targets the real limit rather than our request.
        self.max_model_len = self._resolve_max_model_len(max_model_len)
        print(f"Effective max_model_len for prompt fitting: {self.max_model_len}")

        # temperature=0 => greedy (deterministic) decoding.
        sampling_kwargs = dict(
            temperature=0.0,
            max_tokens=max_gen_tokens,
        )
        if guided_json:
            # vLLM >=0.24 renamed GuidedDecodingParams -> StructuredOutputsParams
            # and the SamplingParams field guided_decoding -> structured_outputs.
            # Prefer the new API, fall back to the old for older vLLM.
            try:
                from vllm.sampling_params import StructuredOutputsParams
                sampling_kwargs['structured_outputs'] = StructuredOutputsParams(
                    json=SPANS_JSON_SCHEMA
                )
            except ImportError:
                from vllm.sampling_params import GuidedDecodingParams
                sampling_kwargs['guided_decoding'] = GuidedDecodingParams(
                    json=SPANS_JSON_SCHEMA
                )
            print("Enabled vLLM guided JSON decoding for the 'spans' schema.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def _format_prompt(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Some chat templates (e.g. Gemma) reject a standalone system role;
            # fall back to prepending the system text to the user turn.
            merged = [{
                "role": "user",
                "content": f"{system_prompt}\n\n{user_prompt}",
            }]
            return self.tokenizer.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True
            )

    def _resolve_max_model_len(self, fallback):
        """Read the context length the vLLM engine actually uses (it can cap the
        requested value). Fall back to the requested value if the attribute path
        differs across vLLM versions."""
        for accessor in (
            lambda: self.llm.llm_engine.model_config.max_model_len,
            lambda: self.llm.llm_engine.vllm_config.model_config.max_model_len,
        ):
            try:
                value = accessor()
                if value:
                    return int(value)
            except Exception:
                pass
        return fallback

    def _ntokens(self, formatted_prompt):
        return len(self.tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"])

    def fit_prompt(self, template, record):
        """Render ``(system, user)`` for one record, truncating ONLY the passage
        (keeping the instructions and query intact) so the formatted prompt plus
        the generation budget fits inside ``max_model_len``. Long documents that
        would otherwise abort the whole run with a context-length error are cut
        down here instead.

        Returns ``(system, user, meta)`` where ``meta`` carries per-sample token
        stats for the output dataset:
          - ``was_truncated``:            whether the passage was cut to fit
          - ``document_tokens``:          passage token count before truncation
          - ``document_tokens_truncated``: passage token count actually sent
          - ``prompt_tokens``:            full formatted prompt (instructions +
                                          query + passage) token count sent
        These reuse the tokenization we already do here, so they add no
        meaningful cost to generation.
        """
        system, user = create_message(template, **record)
        prompt_tokens = self._ntokens(self._format_prompt(system, user))

        has_psg = self.psg_key in record
        psg_ids = (self.tokenizer(record[self.psg_key], add_special_tokens=False)["input_ids"]
                   if has_psg else [])
        doc_tokens_full = len(psg_ids) if has_psg else None

        def meta(was_truncated, doc_tokens_sent):
            return {
                "was_truncated": was_truncated,
                "document_tokens": doc_tokens_full,
                "document_tokens_truncated": doc_tokens_sent,
                "prompt_tokens": prompt_tokens,
            }

        budget = (self.max_model_len - self.max_gen_tokens - self.TRUNC_MARGIN
                  if self.max_model_len else None)
        if budget is None or prompt_tokens <= budget or not has_psg:
            return system, user, meta(False, doc_tokens_full)

        # Trim the passage token count by (overflow + margin), then re-render and
        # re-check a few times to converge despite tokenizer boundary effects.
        keep = max(0, doc_tokens_full - (prompt_tokens - budget) - self.TRUNC_MARGIN)
        for _ in range(5):
            trimmed = dict(record)
            trimmed[self.psg_key] = self.tokenizer.decode(psg_ids[:keep])
            system, user = create_message(template, **trimmed)
            prompt_tokens = self._ntokens(self._format_prompt(system, user))
            if prompt_tokens <= budget or keep <= 0:
                break
            keep = max(0, keep - (prompt_tokens - budget) - self.TRUNC_MARGIN)

        self._n_truncated += 1
        return system, user, meta(True, keep)

    def generate(self, prompts):
        """Batched generation. ``prompts`` is a list of (system, user) pairs;
        returns raw content strings aligned to it."""
        formatted = [self._format_prompt(system, user) for system, user in prompts]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


def template_base_path(template_file):
    """Strip a trailing ``.template`` and any ``-system``/``-user`` suffix to get
    the shared base path, so that ``template_name.template``,
    ``template_name-system.template`` and ``template_name-user.template`` all
    resolve to the same base ``template_name``."""
    return re.sub(r'(-system|-user)?\.template$', '', template_file)


def read_system_user_templates(template_file):
    """Always load BOTH the system and the user template for a given base.

    Returns a ``(system_template, user_template)`` pair of jinja2 Templates.
    Both files must exist -- there is no single-template fallback.
    """
    base = template_base_path(template_file)
    with open(f"{base}-system.template", 'r') as f:
        system_template = Template(f.read())
    with open(f"{base}-user.template", 'r') as f:
        user_template = Template(f.read())
    return system_template, user_template


def create_message(templates, **kwargs):
    """Render the (system, user) template pair for one record."""
    system_template, user_template = templates
    return system_template.render(**kwargs), user_template.render(**kwargs)


def task_from_prompt(custom_id, prompt):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": prompt
    }


def messages_for_passages(template, openai_api, **kwargs):
    system_prompt, user_prompt = create_message(template, **kwargs)
    api_message = openai_api.create_api_call_dict(system_prompt, user_prompt)
    return api_message


def create_batch_name(id_from, id_to, generated_data_dir):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    return f"{generated_data_dir}/{time_str}_batch-{id_from}-{id_to}.jsonl"


def create_batch_fix_name(fix_id, generated_data_dir):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    return f"{generated_data_dir}/{time_str}_batch_{fix_id}.jsonl"


def create_batch_file(data_chunk, api, jsonl_filename, template):
    """
    Create batch file for a range of ids in a jsonl format, which is suitable for OpenAI API.
    https://platform.openai.com/docs/guides/batch/getting-started
    """
    with jsonlines.open(jsonl_filename, "w") as task_writer:
        for row_id, record in data_chunk.items():
            message = messages_for_passages(template, api, **record)

            # Create sub-batch
            task = task_from_prompt(f"row_{row_id}", message)
            task_writer.write(task)

    print(f"Batch file saved to {jsonl_filename}")
    return jsonl_filename


def create_batch_job(batch_filename):
    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(batch_filename, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"job on filename {batch_filename}",
            "for_filename": batch_filename
        }
    )

    print(f"Created batch ({batch.id}):")
    print(batch)
    return batch.id


def download_output_batch(batch_id):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    while batch.status != "completed":
        # print(batch)
        print(f"Batch is not completed yet - status is {batch.status}")
        if batch.status == "failed":
            raise Exception(f"Batch failed: {batch}")
        sleep_with_progress(15,
                            description="Waiting for validation")
        batch = client.batches.retrieve(batch_id)

    print("Batch is completed")
    print(batch)
    result_file_id = batch.output_file_id
    result = client.files.content(result_file_id).content
    batch_filename = batch.metadata["for_filename"].replace('.jsonl', '_output.jsonl')
    with open(batch_filename, 'wb') as file:
        file.write(result)
    print(f"Output file saved to {batch_filename}")

    return batch_filename


def message_from_output(res):
    message_json = res['response']['body']['choices'][0]['message']['content']
    try:
        message = json.loads(message_json)
    except json.JSONDecodeError:
        # Remove code block markdown (generated response is not json_obj but text)
        message = message_json.strip("```").strip("json\n")
        message = json.loads(message)
    return message


def sleep_with_progress(seconds, description=None):
    if description is None:
        description = "Waiting"

    for _ in tqdm(range(seconds), desc=description, unit="s"):
        time.sleep(1)


def read_input_data(input_name, from_sample, to_sample, hf_split='train', hf_config=None):
    """Load input records, deciding the source from ``input_name``:
    the LongEmbed dataset gets its dedicated query/passage join, a
    ``.json``/``.jsonl`` path is read from disk, and anything else is treated as
    a HuggingFace dataset id loaded with ``datasets.load_dataset``."""
    if is_longembed(input_name):
        records = load_longembed()
        end = to_sample if to_sample >= 0 else len(records)
        return records[from_sample:end]
    elif input_name.endswith('.jsonl'):
        with jsonlines.open(input_name, 'r') as f:
            return [
                line_obj
                for line_id, line_obj in enumerate(f)
                if from_sample <= line_id < to_sample
            ]
    elif input_name.endswith('.json'):
        with open(input_name, 'r') as f:
            data = json.load(f)
            return data[from_sample:to_sample]
    else:
        from datasets import load_dataset
        dataset = load_dataset(input_name, hf_config, split=hf_split)
        records = [dict(row) for row in dataset]
        return records[from_sample:to_sample]


def silent_remove(output_data_file):
    try:
        os.remove(output_data_file)
    except OSError:
        pass


def process_output(process_file):
    responses = {}
    with open(process_file) as in_file:
        for line in in_file:
            parsed_data = json.loads(line)
            row_index = int(parsed_data['custom_id'].strip("row_"))
            content = parsed_data['response']['body']['choices'][0]['message']['content']
            responses[row_index] = content

    return responses


def get_sorted_generation_files(generated_data_dir):
    """
    Loads files from the directory and sorts them based on:
        batch for fix:        2025-02-02T14:58:08_batch_0_output.jsonl (0 here)
        batch for generation: 2025-02-02T14:58:08_batch-100-200_output.jsonl (100 here)

    :param generated_data_dir: Directory path for fix/generation files
    :return: Sorted list of files to
    """

    files_to_process = []
    for filename in os.listdir(generated_data_dir):
        if filename.endswith("_output.jsonl"):
            process_filename = f"{generated_data_dir}/{filename}"
            files_to_process.append(process_filename)

    # first number after batch- or batch_ is used for sorting
    pattern = r"batch[_-](\d+)"
    return sorted(files_to_process, key=lambda x: int(re.search(pattern, x.split("/")[-1]).group(1)))


def get_all_responses(generated_data_dir):
    """
    :return: Sorted reposes based on custom id
    """
    # row_id -> index, content -> value
    responses = {}

    files_to_process = get_sorted_generation_files(generated_data_dir)
    if not files_to_process:
        print(f"Warning: No output files found to process. in {generated_data_dir}")

    for process_filename in tqdm(files_to_process, desc=f"Processing output files in {generated_data_dir}"):
        responses.update(process_output(process_filename))

    json_decode_error = 0

    def decode_one(v):
        if type(v) is str:
            try:
                object = json.loads(v)
                if 'spans' in object:
                    return object
            except json.JSONDecodeError:
                nonlocal json_decode_error
                json_decode_error += 1
        return {'spans': []}

    # Create new list by sorting with keys - rowid, needed to match input
    outs = {k: decode_one(v) for k, v in responses.items()}
    print(f"JSON decode error count: {json_decode_error}")
    return outs


def generate_one_batch(data_chunk, generation_api, jsonl_filename, generation_client, template):
    if generation_client == 'openai':
        generate_one_batch_openai(data_chunk, generation_api, jsonl_filename, template)
    elif generation_client == 'vllm':
        generate_one_batch_vllm(data_chunk, generation_api, jsonl_filename, template)
    else:
        generate_one_batch_ollama(data_chunk, generation_api, jsonl_filename, template)


def write_batch_output(jsonl_filename, row_choices):
    """Write ``(row_id, choice)`` pairs to ``<jsonl_filename>``'s ``*_output.jsonl``
    counterpart in the schema ``process_output`` reads back
    (``custom_id`` + ``response.body.choices[0]``). Shared by every online client."""
    responses = [
        {
            'custom_id': f"row_{row_id}",
            'response': {'body': {'choices': [choice]}},
        }
        for row_id, choice in row_choices
    ]
    out_filename = jsonl_filename.replace('.jsonl', '_output.jsonl')
    with jsonlines.open(out_filename, mode='w') as writer:
        writer.write_all(responses)


def generate_one_batch_vllm(data_chunk, generation_api, jsonl_filename, template):
    """Generate the whole chunk in a single batched vLLM call."""
    row_ids = list(data_chunk.keys())
    # fit_prompt truncates over-long passages so a single long document can't
    # abort the batch with a context-length error, and reports per-sample token
    # stats. We stash those stats on the input record itself: it is the same
    # object write_output later splats into the final row (see create_minibatch),
    # so was_truncated / *_tokens land in the output dataset with no extra pass.
    prompts = []
    for row_id in row_ids:
        system, user, meta = generation_api.fit_prompt(template, data_chunk[row_id])
        prompts.append((system, user))
        data_chunk[row_id].update(meta)
    contents = generation_api.generate(prompts)
    write_batch_output(jsonl_filename, (
        (row_id, {'message': {'content': content}})
        for row_id, content in zip(row_ids, contents)
    ))


def generate_one_batch_openai(data_chunk, generation_api, jsonl_filename, template):
    create_batch_file(data_chunk,
                      generation_api,
                      jsonl_filename,
                      template
                      )
    batch_id = create_batch_job(jsonl_filename)
    download_output_batch(batch_id)

    sleep_with_progress(60,
                        description="Waiting before sending new batch file.")


def generate_one_batch_ollama(data_chunk, generation_api, jsonl_filename, template):
    """Generate the chunk one record at a time against an OpenAI-compatible server."""
    row_choices = []
    for row_id, record in tqdm(data_chunk.items(), desc="Generating batch"):
        response = generation_api(*create_message(template, **record))
        choice = dict(response.choices[0])
        choice['message'] = dict(choice['message'])
        row_choices.append((row_id, choice))

    write_batch_output(jsonl_filename, row_choices)


def generate_all_batches(data_chunks, generation_api, generated_data_dir, generation_client, template):
    for data_chunk in data_chunks:
        loop_from = list(data_chunk.keys())[0]
        loop_to = list(data_chunk.keys())[-1]
        print(f"Processing {loop_from} to {loop_to}")
        jsonl_filename = create_batch_name(loop_from, loop_to, generated_data_dir)

        generate_one_batch(data_chunk, generation_api, jsonl_filename, generation_client, template)


def generate_all_batches_fix(data_chunks, generation_api, generated_data_dir, generation_client, template):
    for fix_id, data_chunk in enumerate(data_chunks):
        print(f"Creating {fix_id} fix batch file.")
        jsonl_filename = create_batch_fix_name(fix_id, generated_data_dir)
        generate_one_batch(data_chunk, generation_api, jsonl_filename, generation_client, template)


def write_output(responses_out, output_data_file, input_data, from_sample):
    silent_remove(output_data_file)
    with jsonlines.open(output_data_file, mode='w') as writer:
        for in_key, out_selected in responses_out.items():
            writer.write(
                {
                    **input_data[in_key - from_sample],
                    'selected_spans': out_selected.get('spans', [])
                }
            )


def read_out_data(output_data_file):
    out_data = []
    with jsonlines.open(output_data_file, mode='r') as reader:
        for line in reader:
            out_data.append(line)
    return out_data


def write_out_data(output_data_file, out_data):
    with jsonlines.open(output_data_file, mode='w') as writer:
        writer.write_all(out_data)


def update_output_spans(responses_with_keys, output_data_file):
    out_data = read_out_data(output_data_file)

    for k, v in responses_with_keys.items():
        out_data[k]['selected_spans'] = v['spans']

    write_out_data(output_data_file, out_data)


def annotate_failed_extraction(output_data_file, indexes_to_remove):
    out_data = read_out_data(output_data_file)

    for remove_idx in indexes_to_remove:
        out_data[remove_idx]['extraction_error'] = True

    write_out_data(output_data_file, out_data)


def find_invalid_samples(output_data_file, last_invalid_indexes, psg_key):
    # This tokenizer only computes offset mappings for span matching -- nothing
    # is fed to a model, so the 512-token XLM-R limit is irrelevant. Override
    # model_max_length so transformers does not print a "Token indices sequence
    # length is longer than ... (N > 512)" warning for every long passage.
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base',
                                              model_max_length=10**9)
    dataset = ExplanationsDataset(output_data_file, tokenizer,
                                  decode_positive_as_list=True,
                                  error_on_invalid=True, psg_key=psg_key)

    failed_indexes = []
    for i in tqdm(range(len(dataset)), desc="Finding invalid inputs", unit="samples"):
        try:
            if last_invalid_indexes and i not in last_invalid_indexes:
                continue
            dataset[i]
        except AssertionError:
            failed_indexes.append(i)
    return failed_indexes


def find_generated_indexes(generated_data_dir):
    indexes = set()
    for filename in os.listdir(generated_data_dir):
        if filename.endswith("_output.jsonl"):
            with jsonlines.open(os.path.join(generated_data_dir, filename)) as in_file:
                for parsed_data in in_file:
                    row_index = int(parsed_data['custom_id'].strip("row_"))
                    indexes.add(row_index)
    return indexes


def prepare_out_dir(generated_data_dir, force_rewrite):
    try:
        os.makedirs(generated_data_dir)
    except FileExistsError:
        empty_dir = not os.listdir(generated_data_dir)
        if empty_dir:
            return set()

        # dir is not empty, find generated files and skip them for generation
        if force_rewrite:
            already_generated = find_generated_indexes(generated_data_dir)
            print(f"Directory {generated_data_dir} already exists.")
            print(f"Found {len(already_generated)} already generated samples, "
                  f"skipping {min(already_generated)}-{max(already_generated)} indexes.")
            return already_generated
        else:
            raise FileExistsError(f"Directory {generated_data_dir} already exists. "
                                  f"Use --force_rewrite to allow writing more samples there.")
    return set()


def dataset_improved(invalid_samples_history, max_regenerate_count, invalid_len):
    return (len(invalid_samples_history) < max_regenerate_count
            or not sum(invalid_samples_history[-max_regenerate_count:]) == max_regenerate_count * invalid_len)


def find_sorted_fix_dirs(generated_data_dir):
    # Find all fix directories in the target directory
    fix_dirs = []
    for item in os.listdir(generated_data_dir):
        existing_fix_path = os.path.join(generated_data_dir, item)
        if os.path.isdir(existing_fix_path) and item.startswith('fix_'):
            fix_dirs.append(existing_fix_path)
    # Apply fixes in the correct order
    fix_dirs = sorted(fix_dirs, key=lambda x: int(x.split('_')[-1]))
    return fix_dirs


def create_minibatch(input_data, generated_ids, from_sample, batch_size, batch_start,
                     input_data_len):
    minibatch = {}
    for data_idx in range(batch_start, min(batch_start + batch_size, input_data_len)):
        if data_idx + from_sample in generated_ids:
            continue
        minibatch[data_idx + from_sample] = input_data[data_idx]
    return minibatch


def create_batched_input(input_data, generated_ids, from_sample, batch_size):
    input_data_len = len(input_data)
    data_chunks = [
        create_minibatch(input_data, generated_ids,
                         from_sample, batch_size, batch_start, input_data_len)
        for batch_start in range(0, input_data_len, batch_size)
    ]
    data_chunks = [chunk for chunk in data_chunks if chunk]
    return data_chunks


def sanitize_model_name(model_name):
    return model_name.replace('/', '~')


def get_args():
    argparse.ArgumentParser(description='Generate explanations for MSMARCO dataset')
    parser = argparse.ArgumentParser()

    # Task args
    parser.add_argument("--skip_generation", action='store_true',
                        help="only processes each output file")
    parser.add_argument("--skip-regeneration", dest="skip_regeneration", action='store_true',
                        help="skip re-generating samples whose extracted spans are not found "
                             "in the source text (still marks them with extraction_error)")
    parser.add_argument("--force_rewrite", action="store_true",
                        help="Disables the check for generating into existing directory.")

    # Data args
    parser.add_argument('--input_data_name', type=str, required=True,
                        help="Input source: a .json/.jsonl file path, or otherwise a "
                             "HuggingFace dataset id (auto-detected from the value).")
    parser.add_argument('--hf_split', type=str, default='train',
                        help="Split to load when --input_data_name is a HuggingFace dataset.")
    parser.add_argument('--hf_config', type=str, default=None,
                        help="Config/subset name when --input_data_name is a HuggingFace dataset.")
    parser.add_argument('--generate_into_dir', type=str, default="data/generated",
                        help="Directory for storing raw LLM batch outputs and their fixes.")
    parser.add_argument('--template_file', type=str, default='templates/long-embed.template',
                        help="Base path to the prompt template. Both '<base>-system.template' "
                             "and '<base>-user.template' are always loaded (e.g. "
                             "templates/long-embed.template -> long-embed-system/user).")
    parser.add_argument('--psg_key', type=str, default='psg_text',
                        help="Key for the passage text in the input data.")

    # Generation setting args
    parser.add_argument("--model_name",
                        type=str, default="gpt-4o-2024-08-06",
                        help="model to use for generation, also used to select folder to process")
    parser.add_argument("--from_sample", type=int, default=0,
                        help="The starting index of the data samples to process.")
    parser.add_argument("--to_sample", type=int, default=-1,
                        help="The ending index of the data samples to process.")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="The number of samples to process in each batch.")
    parser.add_argument("--max_fixes", type=int, default=-1,
                        help="Maximum number of fixes to apply. (-1 for unlimited)")

    # Generation API args
    parser.add_argument("--generation_client", type=str, choices=['ollama', 'vllm', 'openai'], default='ollama',
                        help="Specify which generation client should be used. "
                             "'vllm' = native offline batched inference (no server, needs a GPU); "
                             "'ollama' = OpenAI-compatible server at OPENAI_BASE_URL; "
                             "'openai' = hosted OpenAI Batch API.")
    parser.add_argument('--api_token_env_var', type=str,
                        default='E_INFRA_API_TOKEN',
                        help='API token for E_INFRA')
    parser.add_argument('--max_retries', type=int, default=50,
                        help="How many times to retry a request that failed with "
                             "APITimeoutError (busy/overloaded server) before crashing. "
                             "Each retry is announced with its count.")

    # Native vLLM offline-inference args (only used with --generation_client vllm)
    parser.add_argument('--vllm_max_model_len', type=int, default=None,
                        help="Max sequence length for the vLLM engine (default: model config).")
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9,
                        help="Fraction of GPU memory vLLM may use.")
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=None,
                        help="Number of GPUs for tensor parallelism "
                             "(default: all visible GPUs).")
    parser.add_argument('--vllm_dtype', type=str, default='auto',
                        help="Model dtype for vLLM (e.g. auto, bfloat16, float16).")
    parser.add_argument('--max_gen_tokens', type=int, default=4096,
                        help="Max tokens to generate per sample (vLLM offline path).")
    parser.add_argument('--vllm_guided_json', action='store_true',
                        help="Constrain vLLM output to the {'spans': [str, ...]} JSON schema.")

    return parser.parse_args()


def main():
    args = get_args()

    # Prepare API
    if not args.skip_generation:
        if args.generation_client == 'vllm':
            generation_api = VLLMGenerator(
                args.model_name,
                max_model_len=args.vllm_max_model_len,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                dtype=args.vllm_dtype,
                max_gen_tokens=args.max_gen_tokens,
                guided_json=args.vllm_guided_json,
                psg_key=args.psg_key,
            )
        else:
            generation_api = OpenAIGenerator(
                args.model_name,
                generation_client=args.generation_client,
                api_token_env_var=args.api_token_env_var,
                max_retries=args.max_retries
            )

    else:
        generation_api = None

    # Always load both the system and the user template for this base.
    template = read_system_user_templates(args.template_file)

    # Read input data
    input_data = read_input_data(args.input_data_name, args.from_sample, args.to_sample,
                                 hf_split=args.hf_split, hf_config=args.hf_config)

    # Prepare out data file
    template_name = os.path.basename(template_base_path(args.template_file))
    model_name = sanitize_model_name(args.model_name)
    batch_dir = f"{model_name}_from{args.from_sample}-to{len(input_data)}"
    generated_data_dir = os.path.join(args.generate_into_dir, template_name, batch_dir)
    if not args.skip_generation:
        # Create output directory and find already generated data if exists
        generated_ids = prepare_out_dir(generated_data_dir, args.force_rewrite)
        data_chunks = create_batched_input(input_data, generated_ids,
                                           args.from_sample, args.batch_size)
        generate_all_batches(data_chunks,
                             generation_api,
                             generated_data_dir,
                             args.generation_client,
                             template
                             )

    responses_out = get_all_responses(generated_data_dir)

    # Remove file if exists
    output_data_file = f"data/extracted_relevancy/{template_name}/{batch_dir}.jsonl"
    os.makedirs(os.path.dirname(output_data_file), exist_ok=True)

    print(f"Saving output data to {output_data_file}")
    write_output(responses_out, output_data_file, input_data, args.from_sample)

    fix_dirs = find_sorted_fix_dirs(generated_data_dir)
    last_fix = 0
    invalid_samples = None
    if not args.skip_regeneration:
        for fix_dir in fix_dirs:
            invalid_samples = find_invalid_samples(output_data_file, invalid_samples, args.psg_key)
            print(f"{last_fix} fixes applied, {len(invalid_samples)} were marked as invalid in the output dataset before exiting")

            responses_out = get_all_responses(fix_dir)
            # assert sorted(list(responses_out.keys())) == sorted(invalid_samples)
            update_output_spans(responses_out, output_data_file)
            last_fix += 1

        print(f"Found {last_fix} fix files.")

        invalid_samples_history = []
        max_regenerate_count = 5
        while (not args.skip_generation
               and (invalid_len := len(
                    invalid_samples := find_invalid_samples(output_data_file, invalid_samples, args.psg_key))) > 0):
            print(f"Version after {last_fix} fixes has {invalid_len} invalid samples. "
                  f"Trying to fix following indexes.")
            print(invalid_samples)

            fix_data_dir = os.path.join(generated_data_dir, f"fix_{last_fix}")
            os.makedirs(fix_data_dir)
            # get chunks of data but only for invalid indexes
            invalid_data_chunks = [
                {invalid_id: input_data[invalid_id] for invalid_id in invalid_samples[i:i + args.batch_size]}
                for i in range(0, len(invalid_samples), args.batch_size)
            ]
            generate_all_batches_fix(invalid_data_chunks,
                                     generation_api,
                                     fix_data_dir,
                                     args.generation_client,
                                     template
                                     )

            # Update final output file with generated fixes
            responses_out = get_all_responses(fix_data_dir)
            update_output_spans(responses_out, output_data_file)
            last_fix += 1

            # Exit loop if N numbers of tries did not improve dataset
            invalid_samples_history.append(invalid_len)
            if not dataset_improved(invalid_samples_history, max_regenerate_count, invalid_len):
                print(f"Exiting because the count of invalid samples "
                      f"was not changed in last {max_regenerate_count} iterations.")
                break

            # Negative = unlimited
            if 0 < args.max_fixes <= last_fix:
                print(f"Exiting because maximum number of fixes reached ({args.max_fixes}).")
                break

    invalid_samples = find_invalid_samples(output_data_file, invalid_samples, args.psg_key)
    annotate_failed_extraction(output_data_file, invalid_samples)
    print(f"{last_fix} fixes applied, {len(invalid_samples)} were marked as invalid in the output dataset before exiting")


if __name__ == "__main__":
    main()
