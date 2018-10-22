import os
import time
import math
import logging
from tqdm import trange
from random import choice

import torch


def train(loader, model, optim, loss_fn, label_map, num_steps, temp_dir):
    """Training process"""
    start = time.time()
    current_loss = 0

    logging.info("\n\nStarting training...")
    model.train()

    for step in trange(num_steps):
        X, y = next(loader.load())

        optim.zero_grad()

        output = model(X)
        loss = loss_fn(output, y)

        loss.backward()
        optim.step()

        current_loss += loss

        if step % 10 == 0:
            pred = top_pred(output)
            correct = "✓" if pred == y else "✗"

            logging.info(
                "Step: {}    Elapsed: {}    Loss: {:.6g}    Pred: {}    Correct: {}".format(
                    step, time_since(start), loss, label_map[pred], correct
                )
            )

    logging.info("Training complete!")
    logging.info("Now saving...\n\n")
    torch.save(model, os.path.join(temp_dir, "saved_model.pt"))


def test(loader, label_map, temp_dir):
    """Testing process"""
    logging.info("Evaluating...")
    model = torch.load(os.path.join(temp_dir, "saved_model.pt"))
    model.eval()

    num = 0
    num_correct = 0
    for step, pair in enumerate(oader.load("test")):
        X, y = pair
        output = model(X)
        pred = top_pred(output)

        correct = "✓" if pred == y else "✗"
        if correct == "✓":
            num_correct += 1

        logging.info(
            "Step: {}    Pred: {}    Correct: {}".format(step, label_map[pred], correct)
        )
        num += 1

    acc = num_correct / num * 100
    logging.info("Test accuracy: {:.6g}".format(acc))
    logging.info("Evaluation complete!")
    return acc


def top_pred(output: torch.tensor):
    """Get top prediction"""
    _, top_i = output.data.topk(1)
    return top_i[0][0].item()


def time_since(since: float) -> str:
    """Calculate time elapsed"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
