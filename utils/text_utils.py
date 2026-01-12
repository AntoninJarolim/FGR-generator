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

