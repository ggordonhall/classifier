import logging
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def to_int(lst: List[str]) -> List[int]:
    """Convert list of strings to ints"""
    try:
        return [int(s) for s in lst]
    except ValueError:
        raise ValueError("All dimensions must be integers!")


def plot_loss(losses, report_every, dir):
    """Plot the training loss.

    Arguments:
        losses {List[float]} -- list of losses
        report_every {int} -- steps
        dir {str} -- dir to save graph
    """

    series = pd.Series(losses)
    rolling = series.rolling(window=(report_every // 5))
    rolling_mean = rolling.mean()
    series.plot()
    rolling_mean.plot(color='red')

    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.grid(True)
    plt.legend(("Training loss", "Running average"))
    plt.savefig("{}/loss_plot.png".format(dir), dpi=300)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    Args:
        log_path {str} -- where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, "w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
