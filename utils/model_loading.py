import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_model_and_tokenizer(model_name='distilbert-base-uncased-distilled-squad', toknizer_model_name="distilbert-base-uncased"):
    def _get_and_save_pretrained_tokenizer(name):
        slow_tokenizer = AutoTokenizer.from_pretrained("%s" % name , return_token_type_ids= True)
        config = AutoConfig.from_pretrained(name)
        if not os.path.exists("%s/" % name):
            os.makedirs("%s/" % name)
            slow_tokenizer.save_pretrained("%s/" % name)
            config.save_pretrained("%s/" % name)

        return slow_tokenizer
    slow_tokenizer = _get_and_save_pretrained_tokenizer(toknizer_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device=device)
    return model, slow_tokenizer