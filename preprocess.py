import logging
from itertools import chain
from typing import List, Dict, Tuple, Any

import spacy


class Text:
    """
    Handle text pre-processing
    """

    def __init__(self, train, test):
        logging.info("Pre-processing data...")

        train_in, train_out = train
        test_in, test_out = test
        train_tokens = tokenise(train_in)
        test_tokens = tokenise(test_in)

        self._vocab = unique(flatten(train_tokens))
        self._word2idx = build_dict(self._vocab)
        self._idx2label, self._label2idx = self._label_dicts(train_out)

        self._train = self._pair(train_tokens, train_out)
        self._test = self._pair(test_tokens, test_out)

    def _pair(self, inp: List[List[str]], out: List[str]) -> List[Tuple[str, int]]:
        """Index and pair input and outputs.

        Arguments:
            inp {List[List[str]]} -- List of lines of tokens
            out {List[str]} -- List of labels

        Returns:
            List[Tuple[List[int], int]] -- List of (List[int], int) pairs
        """

        inp_idx = [indexer(line, self._word2idx) for line in inp]
        out_idx = indexer(out, self._label2idx)
        return list(zip(inp_idx, out_idx))

    def _label_dicts(self, labels: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Build index-to-label and label-to-index dicts.

        Arguments:
            col {List[str]} -- a list of labels

        Returns:
            Tuple[Dict[int, str], Dict[str, int]] --
                Index-to-label and label-to-index dicts
        """
        label2idx = build_dict(set(labels))
        idx2label = {}
        for k, v in label2idx.items():
            idx2label[v] = k
        return idx2label, label2idx

    @property
    def vocab(self):
        return self._vocab

    @property
    def idx2label(self):
        return self._idx2label

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test


def tokenise(text: List[str]) -> List[List[str]]:
    """Spacy tokenise each line of text in dataset"""
    nlp = spacy.load("en")
    return [[word.lower_ for word in nlp(line)] for line in text]


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists"""
    return list(chain.from_iterable(list_of_lists))


def unique(lines: List[str]) -> List[str]:
    """Get unique words from a list of strings"""
    return list(chain(*(line.lower().split() for line in lines if line)))


def indexer(lst: List[Any], lookup: Dict[Any, int]) -> List[int]:
    """Get indices of elements in list"""
    return [lookup.get(x, -1) for x in lst]


def build_dict(unique: List[str]) -> Dict[str, int]:
    """Build string-to-index mapping"""
    d = {}
    for idx, tok in enumerate(unique):
        d[tok] = idx
    return d

