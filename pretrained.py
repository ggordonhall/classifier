import os
import pickle
from typing import Dict

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
            {Dict[str, torch.Tensor]} -- dict of str to embedding
        """

        if not os.path.exists(self._save_path):
            self._build_dict()
        print("Loading pretrained embeddings...")
        return dict(pickle.load(open(self._save_path, "rb")))

    def _build_dict(self):
        """
        Iterate through pretrained embedding and build word to 
        tensor dict.
        """

        print("Building pre-trained embedding...")
        word2tensor = {}
        with open(self._glove, "rb") as f:
            for line in f:
                word, *vect = line.split()
                vect = list(map(float, vect))
                word2tensor[word] = torch.FloatTensor(vect)
        print("Finished building embedding...")
        self._save(word2tensor)

    def _save(self, d):
        """
        Serialise and save embedding dict.

        Arguments:
            d {Dict[str, torch.Tensor]} -- an embedding dict
        """

        pickle.dump(d, open(self._save_path, "wb"))
