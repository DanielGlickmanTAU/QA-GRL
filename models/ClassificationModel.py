import pytorch_lightning as pl
from dataclasses import dataclass
import torch.functional as F
from experiments.TaskParams import TaskParams
from utils.datasets_loading import get_race_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification
import torch

batch_size = 2
num_workers = 2


# @dataclass
class ClassificationModel(pl.LightningModule):
    def __init__(self, model, tokenizer, hparams={}):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(batch)
        # I think it is this loss
        # I think it works that way
        # todo check this hsit
        loss = torch.nn.BCEWithLogitsLoss()(outputs, y)
        # loss = outputs[0]
        # return {'loss': loss, 'log': {'train_loss': loss}}
        return loss

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return torch.optim.Adam(self.parameters(), lr=2e-5)

    def get_data_loader(self, split):
        pass
