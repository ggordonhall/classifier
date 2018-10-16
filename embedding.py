import os
import pickle
import logging
from typing import Dict

import numpy as np
import torch


class Glove:
    """
    Load pre-trained Glove embeddings into a dict of PyTorch Tensors.
    """

    def __init__(self, path: str):
        self._glove = os.path.join(os.getcwd(), path)
        self._save_path = self._glove.replace("txt", "pkl")

    def load(self):
        """
        Load pretrained embedding dict.

        Returns:
            {Dict[str,  np.array]} -- dict of str to embedding
        """

        if not os.path.exists(self._save_path):
            self._build_dict()
        logging.info("Loading pretrained embeddings...")
        return dict(pickle.load(open(self._save_path, "rb")))

    def _build_dict(self):
        """
        Iterate through pretrained embedding and build word to 
        vector dict.
        """

        logging.info("Building pre-trained embedding...")
        word2vect = {}
        with open(self._glove, "rb") as f:
            for line in f:
                word, *vect = line.split()
                word2vect[word] = np.array(vect).astype(np.float)
        logging.info("Finished building embedding...")
        self._save(word2vect)

    def _save(self, d):
        """
        Serialise and save embedding dict.

        Arguments:
            d {Dict[str, np.array]} -- an embedding dict
        """

        pickle.dump(d, open(self._save_path, "wb"))


def build_embedding(pretrained, emb_dim, vocab):
    """
    Build PyTorch embedding layer with pretrained embeddings.

    Arguments:
        pretrained {Dict[str, np.array]} -- words to pretrained embeddings map
        emb_dim {int} -- the dimension of embedding vectors
        vocab {List[str]} -- list of unique words

    Returns:
        nn.Embedding -- a PyTorch embedding layer
    """

    return embedding_layer(weights(pretrained, emb_dim, vocab))


def weights(pretrained, emb_dim, vocab):
    """
    Build a matrix of embedding weights for vocab with pretrained embeddings.

    Arguments:
        pretrained {Dict[str, np.array]} -- words to pretrained embeddings map
        emb_dim {int} -- the dimension of embedding vectors
        vocab {List[str]} -- list of unique words

    Returns:
        np.array -- an embedding matrix
    """

    matrix = np.zeros((len(vocab), emb_dim))
    for idx, word in enumerate(vocab):
        try:
            matrix[idx] = pretrained[str.encode(word)]
        except KeyError:
            matrix[idx] = np.random.randn(emb_dim)
    return matrix


def embedding_layer(matrix, trainable=True):
    """
    Load a weights matrix into an nn.Embedding layer.

    Arguments:
        matrix {np.array} -- a matrix of weights

    Keyword Arguments:
        trainable {bool} -- trainable weights? (default: {True})

    Returns:
        nn.Embedding -- a PyTorch embedding layer
    """

    weights = torch.from_numpy(matrix)
    num_embs, emb_dim = weights.size()
    emb_layer = torch.nn.Embedding(num_embs, emb_dim)
    emb_layer.load_state_dict({"weight": weights})
    if not trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer
