from dataclasses import dataclass, asdict, is_dataclass
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __str__(self): return str(asdict(self)) if is_dataclass(self) else super().__str__()

    def __repr__(self): return str(asdict(self)) if is_dataclass(self) else super().__repr__()


@dataclass(repr=False)
class _model_params(AttrDict):
    model_name: str
    model_tokenizer: str
    batch_size: int
    learning_rate: float
    num_epochs: int = 4

    def clone(self):
        return _model_params(self.model_name, self.model_tokenizer, self.batch_size, self.learning_rate,
                             self.num_epochs)


_distilbert_squad = _model_params('distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased', 32, 3e-5)
_roberta_squad = _model_params('deepset/roberta-base-squad2', 'roberta-base', 12, 1e-5)
_electra_squad = _model_params('ahotrod/electra_large_discriminator_squad2_512',
                               'ahotrod/electra_large_discriminator_squad2_512', 2, 1e-5)
_t5_qg_small = _model_params('valhalla/t5-small-e2e-qg', 't5-small', 12, 1e-4, num_epochs=40)
_t5_qg_base = _model_params('valhalla/t5-base-e2e-qg', 't5-base', 6, 1e-5, num_epochs=3)


@dataclass(repr=False)
class _race(AttrDict):
    negative_samples_per_question: int = 1


hyperparams = AttrDict()
race = _race()
hyperparams.env = 'UNI' if 'HOST' in os.environ and 'gamir' in os.environ[
    'HOST'] else 'LOCAL' if 'USERNAME' in os.environ else 'AWS'
hyperparams.task_name = 'boolq'
hyperparams.model_params = _roberta_squad
hyperparams.model_params = _t5_qg_base

hyperparams.race = race
hyperparams.use_unique_seperator_for_answer = False
hyperparams.return_overflowing_tokens = False
hyperparams.disable_tqdm = False
print('using hyperparams:', hyperparams)

beam_search_args = {
    "max_length": 512,
    "num_beams": 12,
    "length_penalty": 1,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "num_return_sequences": 2
}

top_k_args = {
    "max_length": 512,
    "do_sample": True,
    "top_k": 50,
    "num_return_sequences": 2
}

top_p_args = {
    "max_length": 512,
    "do_sample": True,
    "top_k": 25,
    'top_p': 0.85,
    "num_return_sequences": 4
}

active_config = beam_search_args
hyperparams.active_config = top_p_args
