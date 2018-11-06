from typing import Union, Iterable
import pandas as pd

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from allennlp.data.vocabulary import Vocabulary

from allennlp.data.dataset import Batch
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import DatasetReader

from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder


class TabularReader(DatasetReader):
    """Read data from a tabular dataset and build source and target
    fields for indexing. First tokenise with the default Allennlp
    tokeniser, ``WordTokenizer()``. Use the ``ELMoTokenCharactersIndexer``
    to index text fields. Label fields undergo default indexing.

    Inherits from the Allennlp ``DatasetReader`` class with the methods
    ``_read()`` and ``text_to_instance()``.

    Arguments:
        text_name {str} -- the input column name
        label_name {str} -- the output column name
        sep {Union["\t", ","]} -- the tabular separator
    """

    def __init__(self, text_name, label_name, sep):
        super().__init__(lazy=False)
        self.sep = sep
        self.text_name = text_name
        self.label_name = label_name
        self.tokeniser = WordTokenizer()
        self.token_indexers = {"character_ids": ELMoTokenCharactersIndexer()}

    def _read(self, file_path):
        """Read the data and yield a tokenised ``Instance`` classes.

        Arguments:
            file_path {str} -- path to tabular data

        Returns:
            {Iterable} -- an ``Instance``
        """

        df = pd.read_csv(file_path, self.sep)
        text_col, label_col = df[self.text_name], df[self.label_name]
        for idx, sentence in enumerate(text_col):
            yield self.text_to_instance(self.tokeniser.tokenize(sentence), label_col[idx])

    def text_to_instance(self, tokens, label):
        """Build text and label field and convert tokens
        to an ``Instance``.

        Arguments:
            tokens {List[str]} -- tokens
            label {str} -- a label

        Returns:
            {Instance} -- a data instance
        """

        sentence_field = TextField(tokens, self.token_indexers)
        label_field = LabelField(label=label)
        fields = {"text": sentence_field, "labels": label_field}
        return Instance(fields)


class ElmoLoader:
    """Read data instances, construct a ``Vocabulary()`` object,
    batch and index the train and test sets. ``ElmoLoader.load()``
    returns an iterator which can be used to generate examples
    ready to be fed to a model. 

    Arguments:
        reader {DatasetReader} -- reader which yields data instances
        train_path {str} -- path to training data
        test_path {str} -- path to the test data
        batch_dims {Tuple[int]} -- [description]
    """

    def __init__(self, reader, train_path, test_path, batch_dims):
        train_batch_dim, test_batch_dim = batch_dims

        train_dataset = reader.read(train_path)
        test_dataset = reader.read(test_path)
        vocab = Vocabulary.from_instances(train_dataset)

        train_iterator = BasicIterator(batch_size=train_batch_dim)
        train_iterator.index_with(vocab)
        test_iterator = BasicIterator(batch_size=test_batch_dim)
        test_iterator.index_with(vocab)

        self._label_map = vocab._index_to_token["labels"]
        self._iterators = {"train": (train_iterator, train_dataset), "test": (
            test_iterator, test_dataset)}

    def load(self, mode="train"):
        """Load input, output pairs from a
        data iterator. Default train, but can
        also load test set.

        Keyword Arguments:
            mode {str} -- [train/test split] (default: {"train"})
        """

        iterator, instances = self._iterators[mode]
        for instance in iterator(instances):
            X = instance["text"]["character_ids"]
            y = instance["labels"]
            yield (X, y)

    @property
    def label_map(self):
        return self._label_map
