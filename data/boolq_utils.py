from data.special_tokens import get_answer_seperator


def remove_speical_tokens(tokenizer, example: str):
    for special in list(tokenizer.special_tokens_map.values()):
        example = example.replace(special, '')
    return example


def get_t_q_a(tokenizer, example):
    def remove_sep_if_starting_with_sep(str):
        return str[len(sep):] if str.startswith(sep) else str

    sep = tokenizer.sep_token

    q_seperator = get_answer_seperator(tokenizer)
    str = tokenizer.decode(example['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    str = remove_sep_if_starting_with_sep(str)
    idx = str.index(sep) + len(sep)

    t = str[:idx]
    str = str[idx:]
    str = remove_sep_if_starting_with_sep(str)
    idx = str.index(q_seperator) + len(q_seperator)

    q = str[:idx]
    a = example['label']

    return remove_speical_tokens(tokenizer, t), remove_speical_tokens(tokenizer, q), a
