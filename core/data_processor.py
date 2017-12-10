import os
import torch
from torch.autograd import Variable
from .vocabulary import Vocabulary
from .common_helpers import *

class DataProcessor(object):
    """Defines the base set of functionality to process a data set into words or characters."""
    def __init__(self, end_of_sequence_token = None, numeric_token = None, unknown_token = None):
        """Initializes an instance of the `DataProcessor` class.

        Parameters
        ----------
        end_of_sequence_token : :obj:`object`, optional
            If provided, the item used to indicate the end of sequence.
        numeric_token : :obj:`object`, optional
            If provided, the item used to indicate a numeric value.
        unknown_token : :obj:`object`, optional
            If provided, the itmem used to indicate an unknown value.
        """
        self._vocabulary = Vocabulary(eos_item = end_of_sequence_token, numeric_item = numeric_token, unknown_item = unknown_token)

    def _create_batches(self, data, batch_size, use_cuda):
        batch_count = data.size(0) // batch_size
        data = data.narrow(0, 0, batch_count * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        if (use_cuda):
            data = data.cuda()
        return data

    def _normalize_words(self, line):
        words = StringHelper.tokenize_words(line) + [self._vocabulary.eos_item()]
        return words

    def _normalize_word_characters(self, word):
        word_characters = None
        if (self._vocabulary.is_eos_item(word)):
            word_characters = [word]
        elif (self._vocabulary.is_numeric_item(word) or self._vocabulary.is_unknown_item(word)):
            word_characters = [word, " "]
        else:
            word_characters = list(word + " ")
        return word_characters

    def _parse_as_word_tokens_iterator(self, string):
        words = self._normalize_words(string)
        for word in words:
            yield word

    def _parse_as_word_tokens(self, string):
        tokens = []
        for word in self._parse_as_word_tokens_iterator(string):
            tokens.append(word)
        return tokens

    def _preprocess_file_as_words(self, file_path):
        iterator = FileHelper.read_lines_iterator(file_path)
        tokens_count = 0
        for line in iterator:
            tokens_iterator = self._parse_as_word_tokens_iterator(line.strip())
            for token in tokens_iterator:
                self._vocabulary.add(token)
                tokens_count += 1
        return tokens_count

    def _process_file_as_words(self, file_path, tokens_count):
        data = torch.LongTensor(tokens_count)
        iterator = FileHelper.read_lines_iterator(file_path)
        token_index = 0
        for line in iterator:
            tokens_iterator = self._parse_as_word_tokens_iterator(line.strip())
            for token in tokens_iterator:
                index = self._vocabulary.index_from_item(token)
                data[token_index] = index
                token_index += 1
        return data

    def _configure_file_as_words(self, file_path):
        tokens_count = self._preprocess_file_as_words(file_path)
        data = self._process_file_as_words(file_path, tokens_count)
        return data

    def _parse_as_character_tokens_iterator(self, string):
        word_tokens_iterator = self._parse_as_word_tokens_iterator(string)
        for word in word_tokens_iterator:
            tokens = self._normalize_word_characters(word)
            for token in tokens:
                yield token

    def _parse_as_character_tokens(self, string):
        tokens = []
        tokens_iterator = self._parse_as_character_tokens_iterator(string)
        for token in tokens_iterator:
            tokens.append(token)
        return tokens

    def _preprocess_file_as_characters(self, file_path):
        iterator = FileHelper.read_lines_iterator(file_path)
        tokens_count = 0
        for line in iterator:
            tokens_iterator = self._parse_as_character_tokens_iterator(line.strip())
            for token in tokens_iterator:
                self._vocabulary.add(token)
                tokens_count += 1
        return tokens_count

    def _process_file_as_characters(self, file_path, tokens_count):
        data = torch.LongTensor(tokens_count)
        iterator = FileHelper.read_lines_iterator(file_path)
        token_index = 0
        for line in iterator:
            tokens_iterator = self._parse_as_character_tokens_iterator(line.strip())
            for token in tokens_iterator:
                index = self._vocabulary.index_from_item(token)
                data[token_index] = index
                token_index += 1
        return data

    def _configure_file_as_characters(self, file_path):
        tokens_count = self._preprocess_file_as_characters(file_path)
        data = self._process_file_as_characters(file_path, tokens_count)
        return data

class PennTreeBankProcessor(DataProcessor):
    """Defines the Penn Tree Bank data processor."""
    def __init__(self, train_batch_size = None, eval_batch_size = None, use_cuda = False):
        """Initializes an instance of the `PennTreeBankProcessor` class.
        Parameters
        ----------
        train_batch_size : :obj:`int`, optional
            The size of the batches to use for training data.
        eval_batch_size : :obj:`int`, optional
            The size of the batches to use for evaluation data.
        use_cuda : :obj:`bool`, optional
            Determines whether or not to use CUDA.
        """
        super().__init__(
            end_of_sequence_token = "<eos>",
            numeric_token = "N",
            unknown_token = "<unk>")
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._use_cuda = use_cuda

    def _batchify(self, corpus):
        if (self._train_batch_size is not None):
            corpus.train = corpus.create_batches(corpus.train, self._train_batch_size, self._use_cuda)
        if (self._eval_batch_size is not None):
            corpus.validation = corpus.create_batches(corpus.validation, self._eval_batch_size, self._use_cuda)
            corpus.test = corpus.create_batches(corpus.test, self._eval_batch_size, self._use_cuda)
        return corpus

    def process_corpus_as_words(self, corpus):
        """Processes a corpus and its data as word items.

        Parameters
        ----------
        corpus : :obj:`Corpus`
            The corpus to process.

        Returns
        -------
        :obj:`Corpus`
            The updated corpus with processed data.       
        """
        corpus.train = super()._configure_file_as_words(corpus.train_file_path)
        corpus.validation = super()._configure_file_as_words(corpus.validation_file_path)
        corpus.test = super()._configure_file_as_words(corpus.test_file_path)
        corpus = self._batchify(corpus)
        corpus.vocabulary = self._vocabulary.copy()
        self._vocabulary.remove_all()
        return corpus

    def process_corpus_as_characters(self, corpus):
        """Processes a corpus and its data as character items.

        Parameters
        ----------
        corpus : :obj:`Corpus`
            The corpus to process.

        Returns
        -------
        :obj:`Corpus`
            The updated corpus with processed data.       
        """
        corpus.train = super()._configure_file_as_characters(corpus.train_file_path)
        corpus.validation = super()._configure_file_as_characters(corpus.validation_file_path)
        corpus.test = super()._configure_file_as_characters(corpus.test_file_path)
        corpus = self._batchify(corpus)
        corpus.vocabulary = self._vocabulary.copy()
        self._vocabulary.remove_all()
        return corpus

    def word_items_from_string(self, string):
        """Parses a string as word items.

        Parameters
        ----------
        string : :obj:`str`
            The string to process.

        Returns
        -------
        :obj:`list` of :obj:`str`
            The list of word items.
        """
        items = super()._parse_as_word_tokens(string)
        return items


    def character_items_from_string(self, string):
        """Parses a string as character items.

        Parameters
        ----------
        string : :obj:`str`
            The string to process.

        Returns
        -------
        :obj:`list` of :obj:`str`
            The list of character items.  
        """
        items = super()._parse_as_character_tokens(string)
        return items
