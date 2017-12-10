import pickle
import random
import torch
from torch.autograd import Variable

class Corpus(object):
    """Defines a `Corpus`.

    Attributes
    ----------
    train_file_path : :obj:`str`
        The path to the train data file.
    train : :obj:`torch.LongTensor`
        The trained data represented as indices from the constructed vocabulary.
    validation_file_path : :obj:`str`
        The path to the validation data file.
    validation : :obj:`torch.LongTensor`
        The validation data represented as indices from the constructed vocabulary.
    test_file_path : :obj:`str`
        The path to the test data file.
    test : :obj:`torch.LongTensor`
        The test data represented as indices from the constructed vocabulary.
    vocabulary : :obj:`Vocabulary`
        The vocabulary constructed from the train, validation and test files.
    """
    def __init__(self, train_file_path, validation_file_path, test_file_path):
        """Initializes an instance of the `Corpus` class.

        Parameters
        ----------
        train_file_path : :obj:`str`
            The path to the train data file.
        validation_file_path : :obj:`str`
            The path to the validation data file.
        test_file_path : :obj:`str`
            The path to the test data file.
        """
        self.train_file_path = train_file_path
        self.train = None
        self.validation_file_path = validation_file_path
        self.validation = None
        self.test_file_path = test_file_path
        self.test = None
        self.vocabulary = None

    @staticmethod
    def load(file_path):
        """Loads the corpus from a file path.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to load the corpus from.

        Returns
        -------
        :obj:`Corpus`
            The loaded corpus.
        """
        with open(file_path, "rb") as input_handle:
            corpus = pickle.load(input_handle)
        return corpus

    def save(self, file_path):
        """Saves the current instance to the provided file path.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to store the corpus.
        """
        with open(file_path, "wb") as output_handle:
            pickle.dump(self, output_handle, protocol = pickle.HIGHEST_PROTOCOL)

    def create_batches(self, data, batch_size, use_cuda):
        """Creates batches from the provided data set.

        Parameters
        ----------
        data : :obj:`torch.LongTensor`
            The batched data to split.
        batch_size : :obj:`int`
            The size of the batch.
        use_cuda : :obj:`bool`
            Determines whether or not to move the data to the GPU.

        Returns
        -------
        :obj:`torch.LongTensor`
            The batched data.
        """
        batch_count = data.size(0) // batch_size
        data = data.narrow(0, 0, batch_count * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        if (use_cuda):
            data = data.cuda()
        return data

    def split_batched_data(self, data, index, sequence_length, use_cuda = False, eval_mode = False):
        """Splits the batched data into input and target data.

        Parameters
        ----------
        data : :obj:`torch.LongTensor`
            The batched data to split.
        index : :obj:`int`
            The offset.
        sequence_length : :obj:`int`
            The length of the sequence.
        use_cuda : :obj:`bool`
            Determines whether or not to move the data to the GPU.
        eval_mode : :obj:`bool`, optional
            Whether or not we are in evaluation mode.

        Returns
        -------
        :obj:`tuple`
            The split data where the first value is the input of type :obj:`torch.autograd.Variable`
            and the second value is the target data of type :obj:`torch.autograd.Variable`.
        """
        length = min(sequence_length, len(data) - 1 - index)
        input_data = Variable(data[index:(index + length)], volatile = eval_mode)
        target_data = Variable(data[(index + 1):(index + 1 + length)].view(-1))
        if (use_cuda):
            input_data = input_data.cuda()
            target_data = target_data.cuda()
        return input_data, target_data

    def _create_batch_with_indices(self, data, indices, batch_size, sequence_length, use_cuda, eval_mode):
        length = min(sequence_length, len(data))
        input_data = torch.LongTensor(batch_size, length)
        target_data = torch.LongTensor(batch_size, length)
        for index in range(batch_size):
            start = indices[index]
            end = start + length + 1
            batch = data[start:end]
            input_data[index] = batch[:-1]
            target_data[index] = batch[1:]
        input_data = Variable(input_data, volatile = eval_mode)
        target_data = Variable(target_data)
        if (use_cuda):
            input_data = input_data.cuda()
            target_data = target_data.cuda()
        return input_data, target_data

    def create_batch(self, data, offset, batch_size, sequence_length, use_cuda = False, eval_mode = False):
        """Create a batch of a given sized composed of a defined length instances starting from a given offset.

        Parameters
        ----------
        data : :obj:`torch.LongTensor`
            The batched data to split.
        offset : :obj:`int`
            The offset to start with.
        batch_size : :obj:`int`
            The size of the batch.
        sequence_length : :obj:`int`
            The length of the sequence.
        use_cuda : :obj:`bool`
            Determines whether or not to move the data to the GPU.
        eval_mode : :obj:`bool`, optional
            Whether or not we are in evaluation mode.

        Returns
        -------
        :obj:`tuple`
            The split data where the first value is the input of type :obj:`torch.autograd.Variable`
            and the second value is the target data of type :obj:`torch.autograd.Variable`.
        """
        length = min(sequence_length, len(data))
        indices = [(offset + (length * index)) for index in range(batch_size)]
        input_data, target_data = self._create_batch_with_indices(
            data,
            indices,
            batch_size,
            sequence_length,
            use_cuda,
            eval_mode)
        return input_data, target_data

    def create_random_batch(self, data, batch_size, sequence_length, use_cuda = False, eval_mode = False):
        """Create a random batch of a given size composed of defined length instances.

        Parameters
        ----------
        data : :obj:`torch.LongTensor`
            The batched data to split.
        batch_size : :obj:`int`
            The size of the batch.
        sequence_length : :obj:`int`
            The length of the sequence.
        use_cuda : :obj:`bool`
            Determines whether or not to move the data to the GPU.
        eval_mode : :obj:`bool`, optional
            Whether or not we are in evaluation mode.

        Returns
        -------
        :obj:`tuple`
            The split data where the first value is the input of type :obj:`torch.autograd.Variable`
            and the second value is the target data of type :obj:`torch.autograd.Variable`.
        """
        length = min(sequence_length, len(data))
        indices = [random.randint(0, len(data) - length - 1) for index in range(batch_size)]
        input_data, target_data = self._create_batch_with_indices(
            data,
            indices,
            batch_size,
            sequence_length,
            use_cuda,
            eval_mode)
        return input_data, target_data
