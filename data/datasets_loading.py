from config.ExperimentVariables import hyperparams
from datasets import load_dataset
from utils import decorators as decorators
import os
import data.special_tokens as special_tokens
import utils.compute as compute

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
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding='max_length')
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
    print('my place is ' + os.getcwd())
    dataset = load_dataset("swag", "regular", data_dir=os.getcwd() + '/.cache', cache_dir=os.getcwd() + '/cache',
                           shuffle=True)
    return preprocess(dataset, tokenizer, preprocess_function_swag)


d = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def preprocess_function_race(examples, tokenizer):
    def answer_letter_to_target_list(letter):
        return [1 if d[letter] == i else 0 for i in range(4)]

    def adjust_negative_samples_ratio(options, labels):
        should_take_negative = True
        options_new = []
        labels_new = []
        for i, (option, label) in enumerate(zip(options, labels)):
            if i % 4 < hyperparams.race.negative_samples_per_question:
                should_take_negative = True

            if label:
                options_new.append(option)
                labels_new.append(label)
            elif should_take_negative:
                should_take_negative = False
                options_new.append(option)
                labels_new.append(label)

        return options_new, labels_new

    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    texts = [[context] * 2 for context in examples["article"]]
    # Grab all second sentences possible for each context.
    questions = [[context] * 2 for context in examples["question"]]
    # Flatten everything
    texts = sum(texts, [])
    questions = sum(questions, [])

    answers = examples['answer']
    if len(examples) == 1: answers = [
        answers]  # make it list so it is iterable..avoids annoying case for single element
    labels = sum([answer_letter_to_target_list(letter) for letter in answers], [])
    options = sum(examples['options'], [])

    options, labels = adjust_negative_samples_ratio(options, labels)

    # Tokenize
    seperator = special_tokens.get_answer_seperator(tokenizer)
    tokenized_examples = tokenizer(texts, [q + seperator + o for q, o in zip(questions, options)],
                                   truncation=True, padding=True,
                                   return_overflowing_tokens=hyperparams.return_overflowing_tokens)

    if hyperparams.return_overflowing_tokens:
        overflown = [x.ids for x in tokenized_examples[:] if len(x.overflowing) > 0]
        if len(overflown) > 1:
            print('OVERFLOWING ANSWER: ', len(overflown), ' Out of: ', len(tokenized_examples[:]))

    return {'input_ids': tokenized_examples['input_ids'], 'attention_mask': tokenized_examples['attention_mask'],
            'label': labels}


def get_race_dataset(tokenizer):
    dataset = load_dataset("race", "middle", cache_dir=compute.get_cache_dir(), shuffle=True)
    return preprocess(dataset, tokenizer, preprocess_function_race)


def get_sst_dataset(tokenizer):
    def preprocess_function(examples, tokenizer):
        return tokenizer(examples["sentence"], truncation=True)

    dataset = load_dataset("glue", "sst2", cache_dir=compute.get_cache_dir(), shuffle=True)
    return preprocess(dataset, tokenizer, preprocess_function=preprocess_function)


def tokenize_boolq(examples, tokenizer):
    tokenized_examples = tokenizer(examples['passage'], examples['question'], truncation=True,
                                   padding=True)
    # Un-flatten
    tags = examples['answer']
    if len(tags) == 1: tags = [tags]  # make it list so it is iterable..avoids annoying case for single element
    labels = [1 if x else 0 for x in tags]

    return {'input_ids': tokenized_examples['input_ids'], 'attention_mask': tokenized_examples['attention_mask'],
            'label': labels}


def get_boolq_dataset(tokenizer, limit=None, remove_duplicates=True):
    print(os.getcwd())
    boolq = load_boolq()
    if remove_duplicates:
        boolq = _remove_duplicate_questions(boolq)

    if limit:
        print('limiting dataset to', limit)
        boolq['train'] = boolq['train'].select(range(10))
        boolq['validation'] = boolq['validation'].select(range(10))

    return preprocess(boolq, tokenizer, preprocess_function=tokenize_boolq)


def load_boolq():
    return load_dataset("boolq", cache_dir=compute.get_cache_dir())


def get_boolq_generation_dataset(tokenizer):
    def _process(examples, tokenizer):
        source_text = ['generate questions: ' + ex for ex in examples['passage']]
        target_text = examples['question']
        return {"source_text": source_text, "target_text": target_text}

    dataset = load_dataset("boolq", cache_dir=compute.get_cache_dir())
    dataset = _remove_duplicate_questions(dataset)
    return preprocess(dataset, tokenizer, preprocess_function=_process)


def _remove_duplicate_questions(boolq):
    original_texts = set(boolq['train']['passage'])
    boolq['validation'] = boolq['validation'].filter(lambda example: example['passage'] not in original_texts)
    return boolq
