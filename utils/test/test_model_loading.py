from unittest import TestCase

from models import model_loading


class Test(TestCase):
    def test_get_last_checkpoint(self):
        assert model_loading._get_last_checkpoint(['checkpoint-123', 'checkpoint-234', 'checkpoint-99']) == 'checkpoint-234'
