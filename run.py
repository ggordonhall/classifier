import time
import math
import logging
from tqdm import trange
from random import choice

import torch


def train(data, model, optim, loss_fn, emb, idx2label, num_steps):
    """Training process"""
    start = time.time()
    current_loss = 0

    logging.info("\n\nStarting training...")
    model.train()

    for step in trange(num_steps):
        inp, gold = choice(data)
        x = torch.unsqueeze(torch.tensor(inp, dtype=torch.long), 0)
        y = torch.tensor([gold], dtype=torch.long)

        optim.zero_grad()

        output = model(x)
        loss = loss_fn(output, y)

        loss.backward()
        optim.step()

        current_loss += loss

        if step % 10 == 0:
            pred = top_pred(output)
            correct = "✓" if pred == gold else "✗"

            logging.info(
                "Step: {}    Elapsed: {}    Loss: {:.6g}    Pred: {}    Correct: {}".format(
                    step, time_since(start), loss, idx2label[pred], correct
                )
            )

    logging.info("Training complete!")
    logging.info("Now saving...\n\n")
    torch.save(model, "temp/saved_model.pt")


def test(data, idx2label):
    """Testing process"""
    logging.info("Evaluating...")
    model = torch.load("temp/saved_model.pt")
    model.eval()

    num_correct = 0
    for step, pair in enumerate(data):
        inp, gold = pair
        inp = [x for x in inp if x != -1]  #  remove unks

        if inp:
            x = torch.unsqueeze(torch.tensor(inp, dtype=torch.long), 0)
            output = model(x)
            pred = top_pred(output)

            correct = "✓" if pred == gold else "✗"
            if correct == "✓":
                num_correct += 1

            logging.info(
                "Step: {}    Pred: {}    Correct: {}".format(
                    step, idx2label[pred], correct
                )
            )

    acc = num_correct / len(data) * 100
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
