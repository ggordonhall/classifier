import os
import time
import math
import logging
from random import randint

from typing import Tuple

import torch


def train(loader, model, optim, loss_fn, label_map, num_steps, report_every, temp_dir):
    """Training routine.

    Arguments:
        loader {loader.DataLoader} -- class containing data generating logic
        model {nn.Module} -- a PyTorch module containing model logic
        optim {torch.optim} -- an optimiser
        loss_fn -- a loss function
        label_map {Dict[int, str]} -- map from index to string label
        num_steps {int} -- the number of training iterations
        report_every {int} -- report every x steps
        temp_dir {str} -- directory to save the model

    Returns:
        {List[float]} -- list of losses
    """

    start = time.time()
    losses = []

    logging.info("\n\nStarting training...")
    model.train()

    for step in range(num_steps):
        X, y = next(iter(loader.load()))
        optim.zero_grad()

        output = model(X)
        loss = loss_fn(output, y)

        loss.backward()
        optim.step()

        if step % report_every == 0:
            losses.append(loss.item())
            preds = top_preds(output)

            res, r_idx = random_result(preds)
            correct = "✓" if res == y[r_idx] else "✗"
            logging.info(
                "Step: {}    Elapsed: {}    Loss: {:.6g}    Pred: {}    Correct: {}".format(
                    step, time_since(start), loss, label_map[res], correct)
            )

    logging.info("Training complete!")
    logging.info("Now saving...\n\n")
    torch.save(model, os.path.join(temp_dir, "saved_model.pt"))
    return losses


def test(loader, label_map, temp_dir):
    """Testing routine.

    Arguments:
        loader {loader.DataLoader} -- class containing data generating logic
        label_map {Dict[int, str]} -- map from index to string label
        temp_dir {str} -- directory to save the model

    Returns:
        {float} -- the accuracy of the model on the test set
    """

    logging.info("Evaluating...")
    model = torch.load(os.path.join(temp_dir, "saved_model.pt"))
    model.eval()

    num = 0
    num_correct = 0
    for pair in loader.load("test"):
        X, y = pair
        output = model(X)
        preds = top_preds(output)

        res, r_idx = random_result(preds)
        correct = "✓" if res == y[r_idx] else "✗"
        if correct == "✓":
            num_correct += 1

        logging.info(
            "Step: {}    Pred: {}    Correct: {}".format(
                num + 1, label_map[res], correct)
        )
        num += 1

    acc = num_correct / num * 100
    logging.info("Test accuracy: {:.6g}".format(acc))
    logging.info("Evaluation complete!")
    return acc


def top_preds(output: torch.tensor) -> torch.tensor:
    """Get top batch predictions"""
    return torch.max(output, 1)[1]


def random_result(preds: torch.tensor) -> Tuple[float, int]:
    """Return random prediction and its index"""
    idx = randint(0, preds.size(0) - 1)
    return preds[idx], idx


def time_since(since: float) -> str:
    """Calculate time elapsed"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
