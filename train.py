import argparse
from pathlib import Path

import pytorch_lightning as pl

from datasets import *
from models.bare import Bare

from utils import generate_bigrams, count_parameters, save_vocab

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lr", default=1e-3, type=float, help="learning rate, default: 1e-3"
)
parser.add_argument("--batch_size", default=8, type=int, help="batch size, default: 8")
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="number of epochs to train for, default: 100",
)
parser.add_argument("--n_gpus", type=int, default=0, help="number of GPUs, default: 0")
parser.add_argument(
    "--optimizer", type=str, default="adam", help="optimizer to be used, default: adam"
)
parser.add_argument(
    "--valid",
    type=float,
    default=0.2,
    help="fraction of data to validation, default: 0.2",
)

parser.add_argument(
    "--vocab_size",
    type=int,
    default=25_000,
    help="vocabulary size of text field, default: 25,000",
)
#parser.add_argument(
    #"--pretrained",
    #type=str,
    #default="glove.6B.100d",
    #help="pretrained vectors to be used for "
    #"Embedding layer, default: "
    #'"glove.6B.100d"',
#)
parser.add_argument(
    "--dropout_rate",
    type=float,
    default=0.5,
    help="dropout rate to avoid overfitting, default: 0.5",
)
parser.add_argument(
    "--embed_dim",
    type=int,
    default=100,
    help="dimensions of embedding layer, default: 100",
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=256,
    help="number of hidden dim for RNN layers, default: 256",
)
parser.add_argument(
    "--num_layers", type=int, default=1, help="number of RNN layers, default: 1"
)
parser.add_argument(
    "--model",
    type=str,
    default="bare",
    help="model architecture to be used for training, default: bare",
    choices=["bare"],
)
parser.add_argument(
    "--conv_out_channels",
    type=int,
    default=100,
    help="Conv layer out channels, default: 100",
)
parser.add_argument(
    "--filter_size", type=int, default=2, help="Default is bigram, set to n-gram"
)
parser.add_argument(
    "--debug",
    type=bool,
    default=False,
    help="run the model in fast_dev_run mode, default: False",
)
parser.add_argument(
    "--overfit_test",
    type=int,
    default=0,
    help="number of batches on which overfit test to be ran, " "default: 0",
)
parser.add_argument(
    "--seed",
    type=int,
    default=69,
    help="seed value for reproducible results, default: 69",
)

# Add further hyper-parameters here.

hparams = parser.parse_args()

if __name__ == "__main__":
    pl.seed_everything(hparams.seed)

    IMDB_dm = IMDBDataModule()

    IMDB_dm.prepare_data()
    IMDB_dm.setup("fit")

    print("Creating Model...")
    if hparams.model == "bare":
        model = Bare(hparams)
    else:
        raise ValueError(
            "The model doesn't exist select among simple | fast | birnn | cnn"
        )
    hparams.parameters = count_parameters(model)
    print(f"The model has {hparams.parameters:,} trainable parameters")

    if hparams.debug:
        trainer = pl.Trainer(fast_dev_run=True, gpus=hparams.n_gpus)
    elif hparams.overfit_test > 0:
        trainer = pl.Trainer(
            overfit_batches=hparams.overfit_test,
            max_epochs=hparams.epochs,
            gpus=hparams.n_gpus,
            precision=16,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=hparams.n_gpus,
            precision=16,
            gradient_clip_val=1,
            deterministic=True,
        )

    print("Training...")
    trainer.fit(model, IMDB_dm)

    if hparams.debug or hparams.overfit_test > 0:
        print("Debug session complete")
    else:
        print('Training Finished.')
        #print("Testing...")
        #trainer.test()
