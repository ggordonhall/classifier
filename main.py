import os
import logging
import argparse
import pandas as pd

import torch
from torch import nn, optim

import run
from models import DAN
from preprocess import Text
from utils import set_logger
from embedding import Glove, build_embedding


def main(args):
    """Main experiment logic"""
    train_path = os.path.join(args.data_dir, "train.{}".format(args.file_type))
    test_path = os.path.join(args.data_dir, "test.{}".format(args.file_type))
    emb_path = os.path.join(args.data_dir, "glove.6B.{}d.txt".format(args.emb_dim))

    sep = "\t" if args.file_type == "tsv" else ","
    train_df = pd.read_csv(train_path, sep=sep)
    test_df = pd.read_csv(test_path, sep=sep)

    train = (train_df["text"], train_df["gold_label_simple"])
    test = (test_df["text"], test_df["gold_label_simple"])

    data = Text(train, test)
    vocab = data.vocab
    idx2label = data.idx2label

    glove = Glove(emb_path).load()
    embedding = build_embedding(glove, args.emb_dim, vocab)

    model = DAN(args.emb_dim, args.hidden_size, len(idx2label), embedding)
    optimiser = optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    run.train(
        data.train, model, optimiser, loss_fn, embedding, idx2label, args.num_steps
    )

    run.test(data.test, idx2label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="data", help="The directory containing data files"
    )
    parser.add_argument(
        "--file_type",
        default="tsv",
        choices=["tsv", "csv"],
        help="The format of the data file",
    )
    parser.add_argument(
        "--emb_dim",
        default=50,
        choices=[50, 300],
        type=int,
        help="The size of the embedding",
    )
    parser.add_argument(
        "--hidden_size", default=30, type=int, help="The size of the hidden layer"
    )
    parser.add_argument("--lr", default=0.005, type=float, help="The learning rate")
    parser.add_argument(
        "--num_steps", default=1000, type=int, help="The number of training steps"
    )

    args = parser.parse_args()

    temp_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    set_logger(os.path.join(temp_dir, "train.log"))

    main(args)

