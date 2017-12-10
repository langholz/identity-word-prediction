import math
import time
import string
import unidecode
import unicodedata
import torch
import torch.nn as nn

class StringHelper(object):
    """Defines a set of common string operations."""
    @staticmethod
    def ascii_from_unicode_transliteration(str):
        """Transliterate unicode character string into an ascii string.

        Parameters
        ----------
        str : :obj:`str`
            A unicode character string to transliterate.

        Returns
        -------
        :obj:`str`
            An ascii string.
        """
        ascii_str = unidecode.unidecode(str)
        return ascii_str

    @staticmethod
    def is_ascii_character(character, ascii_list = string.ascii_letters):
        """Determines whether or not a provided character is ascii.

        Parameters
        ----------
        character : :obj:`str`
            A character to validate.
        ascii_list : :obj:`str`, optional
            A string containing all the characters which are considered valid ascii values.
            By default we include all lower and upper case letters from a to z. (string.ascii_letters)

        Returns
        -------
        :obj:`bool`
            True if ascii, False otherwise.
        """
        is_ascii = (unicodedata.category(character) != "Mn") and (character in ascii_list)
        return is_ascii

    @staticmethod
    def ascii_from_unicode(str, ascii_list = string.ascii_letters):
        """Converts a unicode character string into an ascii string.

        Parameters
        ----------
        str : :obj:`str`
            A unicode character string.
        ascii_list : :obj:`str`, optional
            A string containing all the characters which are considered valid ascii values.
            By default we include all lower and upper case letters from a to z. (string.ascii_letters)

        Returns
        -------
        :obj:`str`
            An ascii string.
        """
        ascii_values = [character for character in unicodedata.normalize("NFD", str)
                                  if StringHelper.is_ascii_character(character, ascii_list)]
        ascii_str = "".join(ascii_values)
        return ascii_str

    @staticmethod
    def tokenize_words(str):
        """Splits a string into words and removes spaces.

        Parameters
        ----------
        str : :obj:`str`
            A string containing words.

        Returns
        -------
        :obj:`list` of :obj:`str`
            The list of words.
        """
        return str.split()

    @staticmethod
    def tokenize_words_iterator(str):
        """Generates words from a string.

        Parameters
        ----------
        str : :obj:`str`
            A string containing words.

        Yields
        ------
        :obj:`str`
            The next word in the string.
        """
        for word in StringHelper.tokenize_words(str):
            yield word

class FileHelper(object):
    """Defines a set of common file operations."""
    @staticmethod
    def read_file(file_path):
        """Reads a file.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to read.

        Returns
        -------
        :obj:`str`
            The file's content.
        """
        file = open(file_path).read()
        return file

    @staticmethod
    def read_file_as_ascii(file_path):
        """Reads a file and transliterates the content to ascii.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to read.

        Returns
        -------
        :obj:`str`
            The file's content as ascii.
        """
        file = read_file(file_path)
        file = StringHelper.ascii_from_unicode_transliteration(file)
        return file

    @staticmethod
    def read_lines_iterator(file_path):
        """Generates lines by reading a file.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to read.

        Yields
        ------
        :obj:`str`
            The next line in the file.
        """
        with open(file_path, "r") as fin:
            for line in fin:
                yield line

    @staticmethod
    def read_lines_as_ascii_iterator(file_path):
        """Generates lines as ascii by reading a file.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to read.

        Yields
        ------
        :obj:`str`
            The next line as ascii in the file.
        """
        for line in read_lines_iterator(file_path):
            yield StringHelper.ascii_from_unicode_transliteration(line)

    @staticmethod
    def read_lines(file_path):
        """Reads a file's lines.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to read.

        Returns
        -------
        :obj:`str`
            The file's lines.
        """
        lines = [line for line in FileHelper.read_lines_iterator(file_path)]
        return lines

    @staticmethod
    def read_lines_as_ascii(file_path):
        """Reads a file's lines as ascii.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to read.

        Returns
        -------
        :obj:`str`
            The file's lines as ascii.
        """
        lines = [line for line in FileHelper.read_lines_as_ascii_iterator(file_path)]
        return lines

class TorchHelper(object):
    """Defines a set of common torch operations."""
    @staticmethod
    def set_seed(seed, use_cuda = False):
        """Sets the seed for generating random numbers"""
        torch.cuda.manual_seed(seed) if (use_cuda) else torch.manual_seed(seed)

    @staticmethod
    def load(file_path, use_cuda = False):
        """Loads an object.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the object to load.
        use_cuda : :obj:`bool`, optional
            Determines whether or not to move the parameters to the GPU.

        Returns
        -------
        :obj:`object`
            The loaded object.
        """
        with open(file_path, "rb") as fin:
            torch_object = torch.load(fin, map_location = lambda storage, loc: storage)
        if (use_cuda):
            torch_object.cuda()
        return torch_object

    @staticmethod
    def save(torch_object, file_path):
        """Saves an object.

        Parameters
        ----------
        torch_object : :obj:`object`
            The object to save.
        file_path : :obj:`str`
            The path to the file to save the object.
        """
        torch.save(torch_object, file_path)

    @staticmethod
    def create_adam_optimizer(parameters, learning_rate):
        """Creates an ADAM optimizer.

        Parameters
        ----------
        parameters : :obj:`iterable`
            The iterable set of parameters to optimize.
        learning_rate : :obj:`float`
            The learning rate.

        Returns
        -------
        :obj:`Adam`
            The constructed ADAM optimizer.
        """
        optimizer = torch.optim.Adam(parameters, lr = learning_rate)
        return optimizer

    @staticmethod
    def create_sgd_optimizer(parameters, learning_rate):
        """Creates an SGD optimizer.

        Parameters
        ----------
        parameters : :obj:`iterable`
            The iterable set of parameters to optimize.
        learning_rate : :obj:`float`
            The learning rate.

        Returns
        -------
        :obj:`SGD`
        """
        optimizer = torch.optim.SGD(parameters, lr = learning_rate)
        return optimizer

    @staticmethod
    def create_cross_entropy_loss():
        """Creates a cross entropy loss cirterion.

        Returns
        -------
        :obj:`torch.nn.modules.loss.CrossEntropyLoss`
            The criterion.
        """
        criterion = nn.CrossEntropyLoss()
        return criterion

    @staticmethod
    def prevent_exploding_gradient(model, clip_grad_max_norm, learning_rate):
        torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad_max_norm)
        for parameter in model.parameters():
            parameter.data.add_(-learning_rate, parameter.grad.data)

class TimeHelper(object):
    """Defines a set of common time operations."""
    @staticmethod
    def time_delta(delta):
        """Calculates the time given the delta.

        Parameters
        ----------
        delta : :obj:`float`
            The delta.

        Returns
        -------
        :obj:`str`
            A string containing the minutes and seconds.
        """
        minutes = math.floor(delta / 60)
        seconds = delta - (minutes * 60)
        return "%dm %ds" % (minutes, seconds)

    @staticmethod
    def time_since(since):
        """Calculates the time since the provided value.

        Parameters
        ----------
        since : :obj:`float`
            The reference time.

        Returns
        -------
        :obj:`str`
            A string containing the minutes and seconds since the reference time. 
        """
        delta = time.time() - since
        time = TimeHelper.time_delta(delta)
        return time
