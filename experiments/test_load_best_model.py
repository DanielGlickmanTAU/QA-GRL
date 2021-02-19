from unittest import TestCase

from utils import compute
from utils import special_tokens
from utils.model_loading import get_model_and_tokenizer_for_classification, get_last_model_and_tokenizer
from config import ExperimentVariables

torch = compute.get_torch()


class Test(TestCase):
    def test_loading_best_models(self):
        model_params = ExperimentVariables._electra_squad
        task = 'race-classification'
        # checkpoint = '../experiments/' + task
        checkpoint = '' + task
        model, tokenizer = get_last_model_and_tokenizer(checkpoint, model_params)

        # todo tokenizer also from path?
        self.assertIsNotNone(model)

        model_params = ExperimentVariables._roberta_squad
        model, tokenizer = get_last_model_and_tokenizer(checkpoint, model_params)

        self.assertIsNotNone(model)

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
