from jinja2 import Template


def tokenize_list(tokens, tokenizer):
    token_list = []
    for t in tokens.tolist():
        token_list.append(tokenizer.decode(t))
    return token_list


def find_span_start_index(text, span: str):
    """
    Find span sub-sequence in text and return text-wise indexes
    """
    if not type(span) is str:
        raise AssertionError("Invalid generated data structure.")

    text_len = len(text)
    span_len = len(span)

    # Loop through possible start indices in `text`
    for i in range(text_len - span_len + 1):
        # Check if the sub-sequence from `text` matches `span`
        if text[i:i + span_len] == span:
            return i  # Return the start index if a match is found

    return -1  # Return -1 if the span is not found in text


def find_spans(text, selected_spans):
    """
    Find spans in the text and return start index, length and end index
    :param text:
    :param selected_spans:
    """
    rationales = []

    if selected_spans is None or len(selected_spans) == 0:
        return rationales

    if not isinstance(selected_spans, list):
        print(selected_spans)
        raise AssertionError("Selected spans must be list!")

    if not isinstance(selected_spans[0], str):
        print(selected_spans)
        raise AssertionError(f"selected_spans[0] must be str but got {type(selected_spans[0])}!")

    for span in selected_spans:
        span_start = find_span_start_index(text, span)
        if span_start == -1:
            raise AssertionError(f"Selected span '{span}' was not found in the text: {text}")
        span_length = len(span)
        rationales.append(
            {
                'start': span_start,
                'length': span_length,
                'end': span_start + span_length,
            }
        )
    return rationales


def find_span(context: str, span: str) -> tuple[int, int]:
    """
    Finds the start and end positions of `span` inside `context`.

    Returns:
        (start_index, end_index)  where end_index is exclusive
        Returns (-1, -1) if span is not found.
    """
    start = context.find(span)
    if start == -1:
        raise AssertionError(f"Span '{span}' was not found in the text: {context}")
    end = start + len(span) - 1
    return start, end


def extract_span(start_str, end_str, generated_text, return_start_end=False):
    """Extract text between start and end tokens."""
    start_idx = generated_text.find(start_str)
    end_idx = generated_text.find(end_str, start_idx + 1)
    ans = ""
    if start_idx != -1 and end_idx != -1:
        ans = generated_text[start_idx + len(start_str):end_idx]

    if return_start_end:
        return ans, start_idx, end_idx

    return ans


def remove_special_token(text, *args, error_on_detection=False):
    for special_token in args:
        if special_token in text:
            if error_on_detection:
                raise AssertionError(f"Special token '{special_token}' cannot in text \n{text}")

            print(f"Warning: Removing special token '{special_token}' from text present in: \n{text}")
            text = text.replace(special_token, "")
    return text


def read_template(path):
    with open(path, 'r') as f:
        template_text = f.read()
    template = Template(template_text)
    template.template_text = template_text
    return template


def get_token_span(tokenizer, text, char_span):
    char_start, char_end = char_span
    encodings = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    token_start = encodings.char_to_token(char_start)
    token_end = encodings.char_to_token(char_end)

    assert token_end is not None
    assert token_start is not None

    return token_start, token_end


def find_first_token(generated_before, generated_ids, tokenizer):
    for i in range(1, len(generated_ids)):
        cur_text = tokenizer.decode(
            generated_ids[:i],
            skip_special_tokens=False,
        )
        if cur_text.startswith(generated_before):
            return i - 1
    return None


def find_last_token(generated_after, generated_ids, tokenizer):
    # Remove EOS from the end for the ends of the strings to be equivalent
    if generated_ids[-1] == tokenizer.eos_token_id:
        generated_ids = generated_ids[:-1]

    for i in range(len(generated_ids) - 1, -1, -1):
        cur_text = tokenizer.decode(
            generated_ids[i:],
            skip_special_tokens=False,
        )
        if cur_text.endswith(generated_after):
            return i
    return None
