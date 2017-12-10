import os
import torch
import unidecode
import re
from nltk import sent_tokenize
from torch.autograd import Variable
from core.vocabulary import Vocabulary
from core.common_helpers import *

class CharDataProcessor(object):
    """Defines the base set of functionality to process a data set into characters."""
    def __init__(self, end_of_sequence_token = None, numeric_token = None, vocabulary = None):
        """Initializes an instance of the `CharDataProcessor` class.

        Parameters
        ----------
        end_of_sequence_token : :obj:`object`, optional
            If provided, the item used to indicate the end of sequence.
        numeric_token : :obj:`object`, optional
            If provided, the item used to indicate a numeric value.
        vocabulary : :obj:`Vocabulary`, optional
            If provided, the vocabulary to use instead of the token values.
        """
        self._vocabulary = vocabulary if (vocabulary is not None) else Vocabulary(eos_item = end_of_sequence_token, numeric_item = numeric_token)

    def _normalize_chars(self, line):
        chars = list(line) + [self._vocabulary.eos_item()]
        return chars

    def _parse_as_tokens_iterator(self, string):
        tokens = self._normalize_chars(string)
        for token in tokens:
            yield token

    def _parse_as_tokens(self, string):
        tokens = []
        for token in self._parse_as_tokens_iterator(string):
            tokens.append(token)
        return tokens

    def _preprocess_file(self, file_path):
        iterator = FileHelper.read_lines_iterator(file_path)
        tokens_count = 0
        for line in iterator:
            tokens_iterator = self._parse_as_tokens_iterator(line.strip())
            for token in tokens_iterator:
                self._vocabulary.add(token)
                tokens_count += 1
        return tokens_count

    def _process_file(self, file_path, tokens_count):
        data = torch.zeros(tokens_count).long()
        iterator = FileHelper.read_lines_iterator(file_path)
        token_index = 0
        for line in iterator:
            tokens_iterator = self._parse_as_tokens_iterator(line.strip())
            for token in tokens_iterator:
                index = self._vocabulary.index_from_item(token)
                data[token_index] = index
                token_index += 1
        return data

    def _configure_file(self, file_path):
        tokens_count = self._preprocess_file(file_path)
        data = self._process_file(file_path, tokens_count)
        return data

    def _preprocess_split_files(self, file_paths):
        token_counts = []
        for file_path in file_paths:
            tokens_count = self._preprocess_file(file_path)
            token_counts.append(tokens_count)
        return token_counts

    def _process_split_files(self, file_paths, token_counts):
        data = torch.zeros(sum(token_counts)).long()
        token_index = 0
        for file_path in file_paths:
            iterator = FileHelper.read_lines_iterator(file_path)
            for line in iterator:
                tokens_iterator = self._parse_as_tokens_iterator(line.strip())
                for token in tokens_iterator:
                    index = self._vocabulary.index_from_item(token)
                    data[token_index] = index
                    token_index += 1
        return data

    def _configure_split_files(self, file_paths):
        token_counts = self._preprocess_split_files(file_paths)
        data = self._process_split_files(file_paths, token_counts)
        return data

class IdentityDataProcessor(CharDataProcessor):
    """Defines the identity data processor."""
    def __init__(self, vocabulary = None, train_batch_size = None, eval_batch_size = None, sequence_length = None, use_cuda = False):
        """Initializes an instance of the `IdentityDataProcessor` class.

        Parameters
        ----------
        vocabulary : :obj:`Vocabulary`, optional
            The vocabulary to use.
        train_batch_size : :obj:`int`, optional
            The size of the batches to use for training data.
        eval_batch_size : :obj:`int`, optional
            The size of the batches to use for evaluation data.
        sequence_length : :obj:`int`, optional
            The sequence length of each instance in the batched data.
        use_cuda : :obj:`bool`, optional
            Determines whether or not to use CUDA.
        """
        super().__init__(end_of_sequence_token = "<eos>", numeric_token = "#", vocabulary = vocabulary)
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._sequence_length = sequence_length
        self._use_cuda = use_cuda

    def _create_batches(self, corpus, data, batch_size, sequence_length, use_cuda, eval_mode):
        batched_data = []
        for index in range(0, data.size(0) - 1, sequence_length * batch_size):
            if ((data.size(0) - index) < (sequence_length * batch_size)):
                break
            input_data, target_data = corpus.create_batch(
                data,
                index,
                batch_size,
                sequence_length,
                use_cuda = use_cuda,
                eval_mode = eval_mode)
            batched_data.append((input_data, target_data))
        return batched_data

    def _batchify(self, corpus):
        if (self._sequence_length is not None):
            if (self._train_batch_size is not None):
                corpus.train = self._create_batches(
                    corpus.train,
                    self._train_batch_size,
                    self._train_batch_size,
                    self._sequence_length,
                    self._use_cuda,
                    False)
            if (self._eval_batch_size is not None):
                corpus.validation = self._create_batches(
                    corpus,
                    corpus.validation,
                    self._eval_batch_size,
                    self._sequence_length,
                    self._use_cuda,
                    True)
                corpus.test = self._create_batches(
                    corpus,
                    corpus.test,
                    self._eval_batch_size,
                    self._sequence_length,
                    self._use_cuda,
                    True)
        return corpus

    def process_corpus(self, corpus):
        """Processes a corpus and its data as items.

        Parameters
        ----------
        corpus : :obj:`Corpus`
            The corpus to process.

        Returns
        -------
        :obj:`Corpus`
            The updated corpus with processed train, validation and test data.
        """
        corpus.train = super()._configure_file(corpus.train_file_path)
        corpus.validation = super()._configure_file(corpus.validation_file_path)
        corpus.test = super()._configure_file(corpus.test_file_path)
        corpus = self._batchify(corpus)
        corpus.vocabulary = self._vocabulary.copy()
        self._vocabulary.remove_all()
        return corpus

    def process_overfit_corpus(self, corpus):
        """Preprocesses a corpus for overfitting by joining the data sets into a single set used as train, validation and test data.

        Parameters
        ----------
        corpus : :obj:`Corpus`
            The corpus to process.

        Returns
        -------
        :obj:`Corpus`
            The updated corpus with processed train, validation and test data.
        """
        file_paths = [corpus.train_file_path, corpus.validation_file_path, corpus.test_file_path]
        corpus.train = super()._configure_split_files(file_paths)
        corpus.validation = super()._configure_split_files(file_paths)
        corpus.test = super()._configure_split_files(file_paths)
        corpus = self._batchify(corpus)
        corpus.vocabulary = self._vocabulary.copy()
        self._vocabulary.remove_all()
        return corpus

    def normalize_sentence(self, sentence):
        """Parses a sentence and normalizes it.

        Parameters
        ----------
        sentence : :obj:`str`
            The sentence to normalize.

        Returns
        -------
        :obj:`str`
            The normalized sentence. 
        """
        sentence = unidecode.unidecode(sentence)
        sentence = re.sub(r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?", r"#", sentence)
        sentence = re.sub(r"[\"-]+", r" ", sentence)
        sentence = re.sub(r"`", r"'", sentence)
        sentence = re.sub(r"[ ]+['][ ]+", r"'", sentence)
        sentence = re.sub(r"[ ]+['](?=[a-zA-Z ])", r" ", sentence)
        sentence = re.sub(r"(?<=[a-zA-Z])['][ ]+", r" ", sentence)
        sentence = re.sub(r"[\.]{2,}", r" ", sentence)
        sentence = re.sub(r"[$%&=|~<>/_\^\[\]{}():;,+*!?]+", r" ", sentence)
        sentence = re.sub(r"[ ]+", r" ", sentence)
        sentence = sentence.strip()
        sentence = re.sub(r"(?<=[a-zA-Z])[']$", r"", sentence)
        sentence = re.sub(r"^['](?=[a-zA-Z])", r"", sentence)
        sentence = re.sub(r"[\.][']$", r"", sentence)
        sentence = re.sub(r"['][\.]$", r"", sentence)
        sentence = re.sub(r"^[ ]", r"", sentence)
        sentence = re.sub(r"[ ]$", r"", sentence)
        sentence = re.sub(r"[\.]$", r"", sentence)
        sentence = sentence.strip()
        return sentence

    def normalize_string(self, string):
        """Parses a string and normalizes it.

        Parameters
        ----------
        string : :obj:`str`
            The string to normalize.

        Returns
        -------
        :obj:`list` of :obj:`str`
            The list of normalized sentences.  
        """
        normalized_sentences = []
        sentences = sent_tokenize(string)
        for sentence in sentences:
            sentence = sentence.strip()
            sentence = self.normalize_sentence(sentence)
            if (sentence != ""):
                normalized_sentences.append(sentence)
        return normalized_sentences

    def items_from_string(self, string):
        """Processes an input string.

        Parameters
        ----------
        string : :obj:`str`
            The string to process.

        Returns
        -------
        :obj:`list` of :obj:`str`
            The list of lists of items.  
        """
        sentences = self.normalize_string(string)
        items = []
        for sentence in sentences:
            tokens = super()._parse_as_tokens(sentence)
            for item in tokens:
                items.append(item)
        return items
