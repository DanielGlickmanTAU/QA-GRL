from data import datasets_loading


class DataProcessor:
    def __init__(self, tokenizer, max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.hl_token = "<hl>"
        self.sep_token = "<sep>"
        self.model_type = 't5'

    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)

        return dataset

    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example

    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example

    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding=True,
            pad_to_max_length=True,
            truncation=True,
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding=True,
            # pad_to_max_length=True,
            truncation=True,
        )

        encodings = {
            'input_ids': source_encoding['input_ids'],
            'labels': target_encoding['input_ids'],
            # 'label': target_encoding['input_ids'], try this?? #TODO
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings


def get_processed_boolq_dataset(tokenizer):
    boolq = datasets_loading.get_boolq_generation_dataset(tokenizer)

    processor = DataProcessor(
        tokenizer,
        max_source_length=256,
        max_target_length=32
    )

    boolq = processor.process(boolq)
    original_texts = set(boolq['train']['source_text'])
    boolq['validation'] = boolq['validation'].filter(lambda example: example['source_text'] not in original_texts)

    # boolq.set_format(type='torch', columns=["source_ids", "target_ids", "attention_mask"])
    # #TODO
    # return boolq.map(lambda examples: {'input_ids':examples})

    #TO DO TENSOR TO TENSOR
    return boolq
