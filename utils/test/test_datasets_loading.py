from unittest import TestCase

from utils.datasets_loading import get_race_dataset, get_swag_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification


class Test(TestCase):
    def test_preprocess_function_race(self):
        def get_t_q_a(tokenizer,example):
            str = tokenizer.decode(example['input_ids'])

            idx = str.index('[SEP]') + 5
            t = str[:idx]
            str = str[idx:]

            idx = str.index('[OPT]') + 5
            q = str[:idx]
            str = str[idx:]

            idx = str.index('[SEP]') + 5
            a = str[:idx]

            return t,q,a

        def ids_to_text(tokenizer,ids):
            return tokenizer.decode(ids)

        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset_race = get_race_dataset(tokenizer)

        r1 = dataset_race['train'][0]
        r2 = dataset_race['train'][1]
        r3 = dataset_race['train'][2]
        r4 = dataset_race['train'][3]


        t1,q1,a1 = get_t_q_a(tokenizer,r1)
        t2,q2,a2 = get_t_q_a(tokenizer,r2)
        t3,q3,a3 = get_t_q_a(tokenizer,r3)
        self.assertEqual(q1,q2)
        #should have 2 samples of each text
        self.assertNotEqual(q2,q3)
        print(r1['label'])
        print(r2['input_ids'])
