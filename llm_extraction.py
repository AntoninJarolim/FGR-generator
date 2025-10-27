import argparse
import json
import os
import time
import re
from datetime import datetime

from jsonlines import jsonlines
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from explainable_dataset import ExplanationsDataset


class OpenAIGenerator:
    def __init__(self, model_name, generation_client=False):
        url = os.getenv('OPENAI_BASE_URL') + '/v1'
        if generation_client == 'ollama':
            print("Initialized OLLAMA generation client.")
            self.client = OpenAI(
                base_url=url,
                api_key='ollama',
            )
        elif generation_client == 'vllm':
            print("Initialized VLLM generation client.")
            self.client = OpenAI(
                base_url=url,
                api_key='EMPTY',
            )
        else:
            self.client = OpenAI()
        self.model = model_name  # self.model = "gpt-4o-2024-08-06" "gpt-4o-mini"

        self.temperature = 0.2
        self.max_tokens = 1024
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.system_message = "You are helpful linguistic specialist eager to complete given task."

    def create_message(self, prompt):
        return [
            {
                "role": "user",
                "content": prompt
            }
        ]

    def create_api_call_dict(self, message, from_prompt=True):
        """
        Create API call for OpenAI API
        :param message: Prompt for the API call or message generated with create_message()
        :param from_prompt: Use to transform user prompt to conversation format
        :return:
        """
        if from_prompt:
            message = self.create_message(message)

        return dict(
            model=self.model,
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            response_format={
                "type": "json_object"
            }
        )

    def __call__(self, message):
        api_dict = self.create_api_call_dict(message)
        generation_result = self.client.chat.completions.create(**api_dict)
        return generation_result


def create_message(query, passage):
    return f""""
    You will be presented with a query and passage, please 
    extract passage spans which are most relevant to the query.

    Span must be comprehensive: removing the span from the text should make it irrelevant 
    to the query.
    Span must be plausible: human reading only this span should be convinced that text is relevant.   
    Let's read both passage and query, and then carefully consider
    relevance of each passage part to the query. 

    You may select multiple spans if needed, but ensure that the selected sections do not overlap. 
    Try not to select entire sentences, but only fine-grained spans.
    Do not correct or modify the text!
    Include all grammatical and syntactic errors from the original text, 
    do not remove senseless spaces or punctuation.

    Return only json_object with key 'spans' and list of selected spans 
    (text, start, end) as value. 
    \n
    Query: {query}
    Passage: {passage}
    """


def task_from_prompt(custom_id, prompt):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": prompt
    }


def messages_for_passages(query, passage, openai_api):
    prompt_str = create_message(query, passage)
    api_message = openai_api.create_api_call_dict(prompt_str)
    return api_message


def create_batch_name(id_from, id_to, generated_data_dir):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    return f"{generated_data_dir}/{time_str}_batch-{id_from}-{id_to}.jsonl"


def create_batch_fix_name(fix_id, generated_data_dir):
    time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    return f"{generated_data_dir}/{time_str}_batch_{fix_id}.jsonl"


def create_batch_file(data_chunk, api, jsonl_filename):
    """
    Create batch file for a range of ids in a jsonl format, which is suitable for OpenAI API.
    https://platform.openai.com/docs/guides/batch/getting-started
    """
    with jsonlines.open(jsonl_filename, "w") as task_writer:
        for row_id, d in data_chunk.items():
            message = messages_for_passages(d["query"], d["positive"], api)

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


def read_input_data(file_path, from_sample, to_sample):
    with jsonlines.open(file_path, 'r') as f:
        return [
            line_obj
            for line_id, line_obj in enumerate(f)
            if from_sample <= line_id < to_sample
        ]


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

    for process_filename in tqdm(files_to_process, desc=f"Processing output files in {generated_data_dir}"):
        responses.update(process_output(process_filename))

    json_decode_error = 0

    def decode_one(v):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            nonlocal json_decode_error
            json_decode_error += 1
            return {'spans': []}

    # Create new list by sorting with keys - rowid, needed to match input
    outs = {k: decode_one(v) for k, v in responses.items()}
    print(f"JSON decode error count: {json_decode_error}")
    return outs


def get_args():
    argparse.ArgumentParser(description='Generate explanations for MSMARCO dataset')
    parser = argparse.ArgumentParser()

    # Task args
    parser.add_argument("--skip_generation", action='store_true',
                        help="only processes each output file")
    parser.add_argument("--force_rewrite", action="store_true",
                        help="Disables the check for generating into existing directory.")

    # Data args
    parser.add_argument('--input_data_name', type=str, required=True,
                        help="Filename to generate relevance extraction data.")
    parser.add_argument('--input_generated_relevance', type=str, required=False, default=None,
                        help="TSV file having extracted relevance, just to not generate twice.")
    parser.add_argument('--generate_into_dir', type=str, required=True,
                        help="Dir where output batches and fixes will be.")

    # Generation setting args
    parser.add_argument("--model_name",
                        type=str, default="gpt-4o-2024-08-06",
                        help="model to use for generation, also used to select folder to process")
    parser.add_argument("--from_sample", type=int, default=0, help="Start index of the batch")
    parser.add_argument("--to_sample", type=int, default=30, help="End index of the batch")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Batching steps: range(-from-sample, --to-sample, --batch-step)")

    # Generation API args
    parser.add_argument("--generation_client", type=str, choices=['ollama', 'vllm', 'openai'], default='ollama',
                        help="Specify which generation client should be used. "
                             "Available: 'ollama', 'vllm', 'openai'")

    return parser.parse_args()


def generate_one_batch(data_chunk, generation_api, jsonl_filename, generation_client):
    if generation_client == 'openai':
        generate_one_batch_openai(data_chunk, generation_api, jsonl_filename)
    else:
        generate_one_batch_ollama(data_chunk, generation_api, jsonl_filename)


def generate_one_batch_openai(data_chunk, generation_api, jsonl_filename):
    create_batch_file(data_chunk,
                      generation_api,
                      jsonl_filename
                      )
    batch_id = create_batch_job(jsonl_filename)
    download_output_batch(batch_id)

    sleep_with_progress(60,
                        description="Waiting before sending new batch file.")


def generate_one_batch_ollama(data_chunk, generation_api, jsonl_filename):
    responses = []
    for row_id, d in tqdm(data_chunk.items(), desc="Generating batch"):
        response = generation_api(create_message(d["q_text"], d["psg_text"]))
        choice = dict(response.choices[0])
        choice['message'] = dict(choice['message'])
        responses.append(
            {
                'custom_id': f"row_{row_id}",
                'response': {
                    'body': {
                        'choices': [choice]
                    }
                }
            }
        )
    out_filename = jsonl_filename.replace('.jsonl', '_output.jsonl')
    with jsonlines.open(out_filename, mode='w') as writer:
        writer.write_all(responses)


def generate_all_batches(data_chunks, generation_api, generated_data_dir, generation_client):
    for data_chunk in data_chunks:
        loop_from = list(data_chunk.keys())[0]
        loop_to = list(data_chunk.keys())[-1]
        print(f"Processing {loop_from} to {loop_to}")
        jsonl_filename = create_batch_name(loop_from, loop_to, generated_data_dir)

        generate_one_batch(data_chunk, generation_api, jsonl_filename, generation_client)


def generate_all_batches_fix(data_chunks, generation_api, generated_data_dir, generation_client):
    for fix_id, data_chunk in enumerate(data_chunks):
        print(f"Creating {fix_id} fix batch file.")
        jsonl_filename = create_batch_fix_name(fix_id, generated_data_dir)
        generate_one_batch(data_chunk, generation_api, jsonl_filename, generation_client)


def write_output(responses_out, output_data_file, input_data, from_sample):
    silent_remove(output_data_file)
    with jsonlines.open(output_data_file, mode='w') as writer:
        for in_key, out_selected in responses_out.items():
            writer.write(
                {
                    **input_data[in_key - from_sample],
                    'selected_spans': out_selected['spans']
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


def find_invalid_samples(output_data_file, last_invalid_indexes):
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    dataset = ExplanationsDataset(output_data_file, tokenizer,
                                  decode_positive_as_list=True,
                                  error_on_invalid=True)

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


def load_already_generated(input_generated_relevance):
    # Key is (q_id, psg_id)
    if input_generated_relevance is None:
        return set()

    already_generated = set()
    with open(input_generated_relevance) as relevanece_file:
        for line in relevanece_file:
            q_id, psg_id, *_ = line.split("\t")
            already_generated.add((int(q_id), int(psg_id)))

    return already_generated


def create_minibatch(input_data, already_generated, generated_ids, from_sample, batch_size, batch_start,
                     input_data_len):
    minibatch = {}
    for data_idx in range(batch_start, min(batch_start + batch_size, input_data_len)):
        if data_idx + from_sample in generated_ids:
            continue
        generate_for = input_data[data_idx]
        q_id, psg_id = generate_for['q_id'], generate_for['psg_id']
        if (q_id, psg_id) not in already_generated:
            minibatch[data_idx + from_sample] = input_data[data_idx]
    return minibatch


def create_batched_input(input_data, already_generated, generated_ids, from_sample, batch_size):
    input_data_len = len(input_data)
    data_chunks = [
        create_minibatch(input_data, already_generated, generated_ids,
                         from_sample, batch_size, batch_start, input_data_len)
        for batch_start in range(0, input_data_len, batch_size)
    ]
    data_chunks = [chunk for chunk in data_chunks if chunk]
    return data_chunks


def main():
    args = get_args()

    # Prepare API
    if not args.skip_generation:
        generation_api = OpenAIGenerator(args.model_name, generation_client=args.generation_client)
    else:
        generation_api = None

    # Read input data
    input_data = read_input_data(args.input_data_name, args.from_sample, args.to_sample)
    already_generated = load_already_generated(args.input_generated_relevance)

    # Prepare out data file
    batch_dir = f"{args.model_name.replace('google/', 'google~')}_from{args.from_sample}-to{args.to_sample}"
    generated_data_dir = os.path.join(args.generate_into_dir, batch_dir)
    if not args.skip_generation:
        # Create output directory and find already generated data if exists
        generated_ids = prepare_out_dir(generated_data_dir, args.force_rewrite)
        data_chunks = create_batched_input(input_data, already_generated, generated_ids,
                                           args.from_sample, args.batch_size)
        generate_all_batches(data_chunks,
                             generation_api,
                             generated_data_dir,
                             args.generation_client
                             )

    responses_out = get_all_responses(generated_data_dir)

    # Remove file if exists
    output_data_file = f"data/extracted_relevancy_outs/{batch_dir}.jsonl"
    print(f"Saving output data to {output_data_file}")
    write_output(responses_out, output_data_file, input_data, args.from_sample)

    fix_dirs = find_sorted_fix_dirs(generated_data_dir)
    last_fix = 0
    invalid_samples = None
    for fix_dir in fix_dirs:
        invalid_samples = find_invalid_samples(output_data_file, invalid_samples)
        print(f"{last_fix} fixes applied, {len(invalid_samples)} were removed from output dataset before exiting")

        responses_out = get_all_responses(fix_dir)
        # assert sorted(list(responses_out.keys())) == sorted(invalid_samples)
        update_output_spans(responses_out, output_data_file)
        last_fix += 1

    print(f"Found {last_fix} fix files.")

    invalid_samples_history = []
    max_regenerate_count = 5
    while (not args.skip_generation
           and (invalid_len := len(invalid_samples := find_invalid_samples(output_data_file, invalid_samples))) > 0):
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
                                 args.generation_client
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

    invalid_samples = find_invalid_samples(output_data_file, invalid_samples)
    annotate_failed_extraction(output_data_file, invalid_samples)
    print(f"{last_fix} fixes applied, {len(invalid_samples)} were removed from output dataset before exiting")


if __name__ == "__main__":
    main()
