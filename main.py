import os
import logging
import argparse
import pandas as pd

import sys

import torch
from torch import nn
from torch import optim
import torch.multiprocessing as mp

import run
from models import DAN
from loader import DataLoader
from utils import set_logger, plot_loss, to_int


def main(args):
    """Experiment logic"""
    # Get file separator and construct paths
    sep = "\t" if args.file_type == "tsv" else ","
    train_path = os.path.join(args.data_dir, "train.{}".format(args.file_type))
    test_path = os.path.join(args.data_dir, "test.{}".format(args.file_type))
    #  Read column headings
    headings = pd.read_csv(train_path, sep=sep, nrows=1).columns
    text, label = "text", "gold_label_{}".format(args.task_type)

    if args.elmo:
        from elmo import TabularReader, ElmoLoader

        # Pretrained urls
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        # Read dataset
        reader = TabularReader(text, label, sep)
        loader = ElmoLoader(reader, train_path, test_path, args.batch_dims)
        # Build model
        label_map = loader.label_map
        embedding_size = 1024
        model = DAN(to_int(args.layers), len(
            label_map), embedding_size=embedding_size, elmo_config=(options_file, weight_file))

    else:
        # Build data loader
        loader = DataLoader(
            args.data_dir,
            args.file_type,
            headings,
            text,
            label,
            to_int(args.batch_dims),
            (args.glove_type, args.glove_dim),
            args.temp_dir,
        )
        # Build model
        vocab, label_map = loader.vocab, loader.label_map
        model = DAN(to_int(args.layers), len(label_map), vocab_size=len(
            vocab), embedding_size=args.glove_dim, pretrained_vecs=vocab.vectors)

    #  Define training functions
    optimiser = optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    logging.info("\n\nStarting training...\n\n")
    if args.num_processes > 1:
        model.share_memory()
        processes = []
        for pid in range(args.num_processes):
            p = mp.Process(target=run.training_process, args=(
                pid, loader, model, optimiser, loss_fn, (args.num_steps // args.num_processes)))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        report_every = int(args.num_steps * 0.01)
        losses = run.train(loader, model, optimiser, loss_fn,
                           label_map, args.num_steps, report_every)
        if args.plot:
            logging.info("\n\nPlotting training schedule...\n\n")
            plot_loss(losses, report_every, args.temp_dir)

    # Save the trained model
    logging.info("\n\nNow saving...\n\n")
    torch.save(model, os.path.join(args.temp_dir, "saved_model.pt"))

    # Test
    model_acc = run.test(loader, label_map, args.temp_dir)

    if args.baseline:
        logging.info(
            "\n\nComparing with multinomial naive bayes baseline...\n\n")
        from bayes import multi_nb

        train, test = pd.read_csv(
            train_path, sep=sep), pd.read_csv(test_path, sep=sep)
        train_txt, test_txt = (
            train[text], train[label]), (test[text], test[label])

        base_acc = multi_nb(train_txt, test_txt)
        logging.info("Model accuracy: {:.6g}".format(model_acc))
        logging.info("Baseline accuracy: {:.6g}".format(base_acc))
        logging.info(
            "{}".format("Model wins!" if model_acc >
                        base_acc else "Baseline wins!")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers",
        nargs="+",
        help="The sizes of the hidden layers (required)",
        required=True,
    )
    parser.add_argument(
        "--data_dir", default="data", help="The directory containing data files"
    )
    parser.add_argument(
        "--temp_dir",
        default="temp",
        help="The directory containing embedding and logging files",
    )
    parser.add_argument(
        "--file_type",
        default="tsv",
        choices=["tsv", "csv"],
        help="The format of the data file",
    )
    parser.add_argument(
        "--task_type",
        default="simple",
        choices=["simple", "extended"],
        help="The complexity of the task",
    )
    parser.add_argument(
        "--elmo",
        action="store_true",
        help="Use pre-trained ELMo embeddings (--glove_type and --glove_dim are void)"
    )
    parser.add_argument(
        "--glove_type",
        default="6B",
        choices=["42B", "840B", "twitter.27B", "6B"],
        help="The type of GloVe embedding (default = 6B)"
    )
    parser.add_argument(
        "--glove_dim",
        default=50,
        choices=[50, 300],
        type=int,
        help="The size of the GloVe embedding",
    )
    parser.add_argument(
        "--batch_dims",
        nargs="+",
        default=(16, 1),
        help="Dimensions of (train, test) data batches (default = (16, 1))",
    )
    parser.add_argument("--lr", default=0.01, type=float,
                        help="The learning rate")
    parser.add_argument(
        "--num_steps", default=1000, type=int, help="The number of training steps"
    )
    parser.add_argument("--num_processes", default=1, type=int,
                        help="Number of parallel training processes (default = 1)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the loss against time")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compare with multinomial naive bayes baseline",
    )

    args = parser.parse_args()

    temp_dir = os.path.join(os.getcwd(), args.temp_dir)
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    set_logger(os.path.join(temp_dir, "train.log"))

    torch.manual_seed(230)

    main(args)
