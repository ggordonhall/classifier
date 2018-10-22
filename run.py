import os
import time
import math
import logging
from random import randint, choice

import torch


def train(loader, model, optim, loss_fn, label_map, num_steps, temp_dir):
    """Training process"""
    start = time.time()
    current_loss = 0

    logging.info("\n\nStarting training...")
    model.train()

    for step in range(num_steps):
        X, y = next(iter(loader.load()))
        optim.zero_grad()

        output = model(X)
        loss = loss_fn(output, y)

        loss.backward()
        optim.step()

        current_loss += loss

        if step % 10 == 0:
            preds = top_preds(output)
            r_idx = randint(0, preds.size(0) - 1)
            correct = "✓" if preds[r_idx] == y[r_idx] else "✗"

            logging.info(
                "Step: {}    Elapsed: {}    Loss: {:.6g}    Pred: {}    Correct: {}".format(
                    step, time_since(
                        start), loss, label_map[preds[r_idx]], correct
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
    for pair in loader.load("test"):
        X, y = pair
        output = model(X)
        preds = top_preds(output)
        r_idx = randint(0, preds.size(0) - 1)

        correct = "✓" if preds[r_idx] == y[r_idx] else "✗"
        if correct == "✓":
            num_correct += 1

        logging.info(
            "Step: {}    Pred: {}    Correct: {}".format(
                num + 1, label_map[preds[r_idx]], correct)
        )
        num += 1

    acc = num_correct / num * 100
    logging.info("Test accuracy: {:.6g}".format(acc))
    logging.info("Evaluation complete!")
    return acc


def top_preds(output: torch.tensor):
    """Get top predictions"""
    return torch.max(output, 1)[1]


def time_since(since: float) -> str:
    """Calculate time elapsed"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
