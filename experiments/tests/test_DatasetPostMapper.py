from unittest import TestCase

from data.DatasetPostMapper import DataSetPostMapper
from utils import compute


class TestDataSetPostMapper(TestCase):
    def test_add_prob_to_be_correct(self):
        torch = compute.get_torch()
        logits = [[10, 0], [10, 0], [1, 1]]
        labels = [1, 0, 1]

        class MockModel():
            def eval(self):
                pass

            def __call__(self, *args, **kwargs):
                class _logits():
                    self.logits = torch.tensor(logits)

                return _logits()

        mapper = DataSetPostMapper(MockModel, None)
        examples = {
            'input_ids': None,
            'attention_mask': None,
            'label': labels
        }

        prob_is_correct = 'prob_is_correct'
        dict_with_prob = mapper.add_prob_to_be_correct(examples, prob_is_correct)
        probs_to_correct = dict_with_prob[prob_is_correct]
        self.assertAlmostEquals(probs_to_correct[0], 0, delta=0.1)
        self.assertAlmostEquals(probs_to_correct[1], 1, delta=0.1)
        self.assertAlmostEquals(probs_to_correct[2], 0.5)
