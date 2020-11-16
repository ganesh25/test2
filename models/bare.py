import torch
from torchtext import data
from torchtext import datasets
from torch import optim
import random
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from utils import bce_loss_with_logits, binary_accuracy



class Bare(pl.LightningModule):
    """
    The bare bone lightning model, any sub-class forward should return output: [b * o]
    """

    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        self.hparams = hparams

        self.embedding = nn.Embedding(
            25000,
            self.hparams.embed_dim,
        )
        self.rnn = nn.RNN(self.hparams.embed_dim, self.hparams.hidden_dim)
        self.fc = nn.Linear(self.hparams.hidden_dim, 1)


    def forward(self, text):
        text, text_lens = text
        print(text)
        embedded = self.embedding(text)
        
        output, hidden = self.rnn(embedded)
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))

    def configure_optimizers(self):
        learning_rate = self.hparams.lr
        optim_name = self.hparams.optimizer

        if optim_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optim_name == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        return optimizer

    def training_step(self, batch, batch_idx):
        examples = batch.text
        labels = batch.label
        logits = self.forward(examples).squeeze(1)
        loss = bce_loss_with_logits(logits, labels)
        acc = binary_accuracy(logits, labels)

        result = pl.TrainResult(loss, checkpoint_on=loss)
        result.log('train_loss', loss, prog_bar=True)
        result.log('train_acc', acc, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        examples = batch.text
        labels = batch.label
        logits = self.forward(examples).squeeze(1)
        loss = bce_loss_with_logits(logits, labels)
        acc = binary_accuracy(logits, labels)

        result = pl.EvalResult()
        result.batch_val_loss = loss
        result.batch_val_acc = acc
        return result

    def test_step(self, batch, batch_idx):
        examples = batch.text
        labels = batch.label
        logits = self.forward(examples).squeeze(1)
        loss = bce_loss_with_logits(logits, labels)
        acc = binary_accuracy(logits, labels)

        result = pl.EvalResult()
        result.batch_test_loss = loss
        result.batch_test_acc = acc
        return result

    def validation_epoch_end(self, validation_step_output_result):
        loss = validation_step_output_result.batch_val_loss.mean()
        acc = validation_step_output_result.batch_val_acc.mean()

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)
        return result

    def test_epoch_end(self, validation_step_output_result):
        avg_loss = validation_step_output_result.batch_test_loss.mean()
        avg_acc = validation_step_output_result.batch_test_acc.mean()

        result = pl.EvalResult()
        result.log('test_loss', avg_loss, prog_bar=True)
        result.log('test_acc', avg_acc, prog_bar=True)
        return result

    def transfer_batch_to_device(self, batch, device):
        text = batch.text[0].to(device)
        text_lens = batch.text[1].to(device)
        batch.text = (text, text_lens)
        batch.label = batch.label.to(device)
        return batch
