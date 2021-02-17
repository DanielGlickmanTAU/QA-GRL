from unittest import TestCase

from utils import compute
from utils import special_tokens
from utils.model_loading import get_model_and_tokenizer_for_classification
from os import listdir

torch = compute.get_torch()


class Test(TestCase):
    def get_last_checkpoint_in_path(self,path):
        files = listdir(path)
        return self.get_last_checkpoint(files)

    def get_last_checkpoint(self, files):
        files = [f for f in files if 'checkpoint-' in f]
        return sorted(files, key=lambda s: int(s[len('checkpoint-'):]))[-1]

    def test_get_last_checkpoint(self):
        assert self.get_last_checkpoint(['checkpoint-123','checkpoint-234','checkpoint-99']) == 'checkpoint-234'

    def test_opt_encoding_is_learned(self):
        model_name = 'deepset/roberta-base-squad2'
        cached_dir = "race-classification" + '/' + model_name + '/checkpoint-53500'
        model_pretrained, tokenizer_pre = get_model_and_tokenizer_for_classification(model_name=model_name)
        model_fine_tuned, tokenizer_fine = get_model_and_tokenizer_for_classification(model_name=cached_dir)

        special_token_tensor = self.tokenize(tokenizer_fine, special_tokens.OPT)
        assert len(special_token_tensor) == 1

        initial_embeddings = self.get_embedding_of_OPT_by_model(model_pretrained, special_token_tensor)
        learned_embedding = self.get_embedding_of_OPT_by_model(model_fine_tuned, special_token_tensor)

        # assert the embedding changed during learning
        assert (learned_embedding - initial_embeddings).norm() > 0.01

        existing_token_tensor = self.tokenize(tokenizer_fine, 'the')

        existing_initial_embeddings = self.get_embedding_of_OPT_by_model(model_pretrained, existing_token_tensor)
        existing_learned_embedding = self.get_embedding_of_OPT_by_model(model_fine_tuned, existing_token_tensor)

        # see that the change in embedding for a new word is indeed more significant than for a word already in vocab.
        assert (learned_embedding - initial_embeddings).norm() > 5 * (
                    existing_initial_embeddings - existing_learned_embedding).norm()

    def tokenize(self, tokenizer_fine, string):
        speical_token_encoded = tokenizer_fine.encode(string)[1:-1]
        special_token_tensor = torch.tensor(speical_token_encoded)
        return special_token_tensor

    def get_embedding_of_OPT_by_model(self, model, special_token_tensor):
        return model.base_model.embeddings.word_embeddings(special_token_tensor.to(compute.get_device()))
