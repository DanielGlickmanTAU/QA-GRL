import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import utils.compute as compute

device = compute.get_device()
print('using device ', device)


def get_model_and_tokenizer_for_qa(model_name='distilbert-base-uncased-distilled-squad',
                                   toknizer_model_name="distilbert-base-uncased"):
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForQuestionAnswering)


def get_model_and_tokenizer_for_classification(model_name='distilbert-base-uncased-distilled-squad',
                                               toknizer_model_name="distilbert-base-uncased"):
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForSequenceClassification)


def _get_model_and_toknizer(model_name, toknizer_model_name, autoModelClass):
    def _get_and_save_pretrained_tokenizer(name):
        # NOTE: token_type_ids, seperates the question segment from text segment(its 0 and 1s array)
        # when using distilbert, it does not return token_type_ids, but the encoder adds [SEP] token
        tokenizer = AutoTokenizer.from_pretrained("%s" % name, return_token_type_ids=True, use_fast=True)
        config = AutoConfig.from_pretrained(name)
        if not os.path.exists("%s/" % name):
            os.makedirs("%s/" % name)
            tokenizer.save_pretrained("%s/" % name)
            config.save_pretrained("%s/" % name)

        return tokenizer

    tokenizer = _get_and_save_pretrained_tokenizer(toknizer_model_name)
    model = autoModelClass.from_pretrained(model_name).to(device=device)
    return model, tokenizer
