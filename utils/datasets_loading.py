from datasets import load_dataset

from utils import decorators as decorators

ending_names = ["ending0", "ending1", "ending2", "ending3"]


def preprocess_function_swag(examples, tokenizer):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in
                        enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    tags = examples['label']
    if len(examples) == 1: tags = [tags]  # make it list so it is iterable..avoids annoying case for single element
    labels = sum([[1 if i == label else 0 for i in range(4)] for label in tags], [])

    return {'input_ids': tokenized_examples['input_ids'], 'attention_mask': tokenized_examples['attention_mask'],
            'label': labels}

    # return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


@decorators.measure_time
def preprocess(dataset, tokenizer, preprocess_function):
    to_remove = list(dataset['train'][0].keys())
    if 'label' in to_remove: to_remove.remove('label')
    return dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True,
                       remove_columns=to_remove)


def get_swag_dataset(tokenizer):
    dataset = load_dataset("swag", "regular")
    return preprocess(dataset, tokenizer, preprocess_function_swag)


d = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def preprocess_function_race(examples, tokenizer):
    def answer_letter_to_target_list(letter):
        return [1 if d[letter] == i else 0 for i in range(4)]

    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    texts = [[context] * 4 for context in examples["article"]]
    # Grab all second sentences possible for each context.
    questions = [[context] * 4 for context in examples["question"]]
    # Flatten everything
    texts = sum(texts, [])
    questions = sum(questions, [])
    options = sum(examples['options'], [])

    # Tokenize
    tokenized_examples = tokenizer(questions, texts, options, truncation=True)
    # Un-flatten
    answers = examples['answer']
    if len(examples) == 1: answers = [
        answers]  # make it list so it is iterable..avoids annoying case for single element
    labels = sum([answer_letter_to_target_list(letter) for letter in answers], [])

    return {'input_ids': tokenized_examples['input_ids'], 'attention_mask': tokenized_examples['attention_mask'],
            'label': labels}


def get_race_dataset(tokenizer):
    dataset = load_dataset("race", "middle")
    return preprocess(dataset, tokenizer, preprocess_function_race)
