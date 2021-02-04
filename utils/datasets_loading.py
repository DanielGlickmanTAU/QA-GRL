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
    if len(examples) == 1: tags = [tags] #make it list so it is iterable..avoids annoying case for single element
    labels = sum([ [1 if i==label else 0 for i in range(4)] for label in tags], [])

    return {'input_ids':tokenized_examples['input_ids'], 'attention_mask':tokenized_examples['attention_mask'], 'label':labels}

    # return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}



@decorators.measure_time
def preprocess_swag(dataset, tokenizer):
    to_remove = list(dataset['train'][0].keys())
    to_remove.remove('label')
    return dataset.map(lambda examples: preprocess_function_swag(examples, tokenizer), batched=True,
                       remove_columns=to_remove)


def get_swag_dataset(tokenizer):
    dataset = load_dataset("swag", "regular")
    return preprocess_swag(dataset, tokenizer=tokenizer)