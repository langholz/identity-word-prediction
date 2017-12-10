import string
import unidecode
import pickle
import torch
from torch.autograd import Variable

class Vocabulary(object):
    """Defines a vocabulary of items of any type.

    Allows the representation and conversion of items for a language model.
    """
    def __init__(self, items = [], item_index_map = {}, eos_item = None, numeric_item = None, unknown_item = None):
        """Initializes an instance of the `Vocabulary` class.

        Parameters
        ----------
        items : :obj:`list`, optional
            The list of items.
        item_index_map : :obj:`dict`, optional
            The item index map in which the keys are the items and the values
            are the corresponding index.
        eos_item : :obj:`object`, optional
            If provided, the item used to indicate the end of sequence.
        numeric_item : :obj:`object`, optional
            If provided, the item used to indicate a numeric value.
        unknown_item : :obj:`object`, optional
            If provided, the itmem used to indicate an unknown value.
        """
        self._items = items
        self._item_index_map = item_index_map
        self._eos_item = eos_item
        self._numeric_item = numeric_item
        self._unknown_item = unknown_item
        if (eos_item is not None):
            self._add_eos_item(eos_item)
        if (numeric_item is not None):
            self._add_numeric_item(numeric_item)
        if (unknown_item is not None):
            self._add_unknown_item(unknown_item)

    @staticmethod
    def load(file_path):
        """Loads the vocabulary from a file path.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to load the vocabulary from.
        """
        with open(file_path, "rb") as input_handle:
            vocabulary = pickle.load(input_handle)
        return vocabulary

    @staticmethod
    def create_item_index_map(items):
        """Creates an index map from a set of items.

        Parameters
        ----------
        items : :obj:`list`
            The list of items to create and index map from.

        Returns
        -------
        dict
            An item index map where the keys are items and the values
            are the corresponding index.
        """
        item_index_map = {}
        for index, item in enumerate(items):
            item_index_map[item] = index
        return item_index_map

    @classmethod
    def from_list(cls, items, eos = None):
        """Creates a Vocabulary instance from a set of items.

        Parameters
        ----------
        items : :obj:`list`
            The list of items to create the vocabulary with.
        eos : :obj:`object`, optional
            If provided, the item used to indicate the end of sequence. 

        Returns
        -------
        Vocabulary
            The vocabulary instance.
        """
        item_index_map = Vocabulary.create_item_index_map(items)
        value = cls(items, item_index_map, eos)
        return value

    @classmethod
    def from_string(cls, str, eos = None):
        """Creates a Vocabulary instance from a string, where each character is an item.

        Parameters
        ----------
        str : str
            The string used to create the vocabulary with.
        eos : :obj:`object`, optional
            If provided, the item used to indicate the end of sequence. 

        Returns
        -------
        Vocabulary
            The vocabulary instance.
        """
        items = list(str)
        item_index_map = ItemDictionary.create_item_index_map(items)
        value = cls(items, item_index_map, eos)
        return value

    def _add_item(self, item):
        self._items.append(item)
        self._item_index_map[item] = len(self._items) - 1

    def _add_eos_item(self, eos_item):
        self.add(eos_item)
        self._eos_item = eos_item

    def _add_numeric_item(self, numeric_item):
        self.add(numeric_item)
        self._numeric_item = numeric_item

    def _add_unknown_item(self, unknown_item):
        self.add(unknown_item)
        self._unknown_item = unknown_item

    def save(self, file_path):
        """Saves the current instance to the provided file path.

        Parameters
        ----------
        file_path : :obj:`str`
            The path to the file to store the vocabulary.
        """
        with open(file_path, "wb") as output_handle:
            pickle.dump(self, output_handle, protocol = pickle.HIGHEST_PROTOCOL)

    def item_exists(self, item):
        """Determines whether or not an item exists in the vocuabulary.

        Parameters
        ----------
        item : :obj:`object`
            The item to validate.

        Returns
        -------
        bool
            True if exists, False otherwise.
        """
        exists = item in self._item_index_map
        return exists

    def add(self, item):
        """Add an item to the vocuabulary.

        Parameters
        ----------
        item : :obj:`object`
            The item to add to the vocabulary.

        Returns
        -------
        bool
            True if exists, False otherwise.
        """
        if (not self.item_exists(item)):
            self._add_item(item)
        return self._item_index_map[item]
    
    def index_from_item(self, item):
        """Retrieves the index given an item.

        Parameters
        ----------
        item : :obj:`object`
            The item to determine its index.

        Returns
        -------
        int
            The index of the item if exists, None otherwise.
        """
        index = self._item_index_map[item]
        return index
    
    def item_from_index(self, index):
        """Retrieves the item given its index.

        Parameters
        ----------
        index : int
            The index of the item.

        Returns
        -------
        :obj:`object`
            The item if exists.
        """
        item = self._items[index]
        return item

    def indices_from_items(self, items):
        """Retrieves the indices given a set of items.

        Parameters
        ----------
        items : :obj:`list`
            The list of items to determine its indices.

        Returns
        -------
        :obj:`list` of :obj:`int`
            The indices of the items that exist. If no item is found None is presented.
        """
        indices = []
        for item in items:
            index = self.index_from_item(item)
            indices.append(index)
        return indices
    
    def items_from_indices(self, indices):
        """Retrieves the items given a set of indices.

        Parameters
        ----------
        indices : :obj:`list` of :obj:`int`
            The list of indices to determine its items.

        Returns
        -------
        :obj:`list`
            The items for the provided indices.
        """
        items = []
        for index in indices:
            item = self.item_from_index(index)
            items.append(item)
        return items

    def is_eos_item(self, item):
        """Determines whether or not the item is an end of sequence item.

        Parameters
        ----------
        item : :obj:`object`
            The item to validate.

        Returns
        -------
        bool
            True if the item is end of sequence item, False otherwise.
        """
        is_eos = False
        if (self._eos_item is not None):
            is_eos = self._eos_item == item
        return is_eos

    def eos_item(self):
        """Defines the end of sequence item.

        Returns
        -------
        :obj:`object`
            The end of sequence item.

        """
        return self._eos_item

    def is_numeric_item(self, item):
        """Determines whether or not the item is a numeric item.

        Parameters
        ----------
        item : :obj:`object`
            The item to validate.

        Returns
        -------
        bool
            True if the item is a numeric item, False otherwise.
        """
        is_numeric = False
        if (self._numeric_item is not None):
            is_numeric = self._numeric_item == item
        return is_numeric

    def numeric_item(self):
        """Defines the numeric item.

        Returns
        -------
        :obj:`object`
            The numeric item.

        """
        return self._numeric_item

    def is_unknown_item(self, item):
        """Determines whether or not the item is an unknown item.

        Parameters
        ----------
        item : :obj:`object`
            The item to validate.

        Returns
        -------
        bool
            True if the item is an unknown item, False otherwise.
        """
        is_unknown = False
        if (self._unknown_item is not None):
            is_unknown = self._unknown_item == item
        return is_unknown

    def unknown_item(self):
        """Defines the unknown item.

        Returns
        -------
        :obj:`object`
            The unknown item.

        """
        return self._unknown_item

    def items(self):
        """The items corresponding to the vocabulary.

        Returns
        -------
        :obj:`list`
            A copy of the items corresponding to the vocabulary.
        """
        return self._items.copy()

    def count(self):
        """The number of items in the vocabulary.

        Returns
        -------
        int
            The item count.
        """
        length = len(self._items)
        return length

    def remove_all(self):
        """Removes all the items and indices.

        Note
        ----
        This does not remove the eos_item, numeric_item or unknown_item.
        """
        self._items = []
        self._item_index_map = {}

    def copy(self):
        """Makes copy of this instance.

        Returns
        -------
        :obj:`Vocabulary`
            The copy of this instance.
        """
        vocabulary = Vocabulary(
            items = self._items.copy(),
            item_index_map = self._item_index_map.copy(),
            eos_item = self.eos_item(),
            numeric_item = self.numeric_item(),
            unknown_item = self.unknown_item())
        return vocabulary

    def long_tensor_from_items(self, items):
        """A collection of items represented as a long tensor.

        Returns
        -------
        :obj:`torch.autograd.variable.Variable` of :obj:`torch.LongTensor`
            The items represented as a long tensor.
        """
        items_length = len(items)
        tensor = torch.zeros(items_length).long()
        indices = self.indices_from_items(items)
        for index in indices:
            tensor[index] = index
        return Variable(tensor)
    
    def one_hot_tensor_from_item(self, item):
        """A collection of items represented as a one hot tensor.

        Returns
        -------
        :obj:`torch.autograd.variable.Variable` of :obj:`torch.FloatTensor`
            The items represented as a one hot tensor.
        """
        tensor = torch.zeros(1, self.count())
        index = self.index_from_item(item)
        tensor[0][index] = 1
        return Variable(tensor)
    
    def one_hot_matrix_tensor_from_items(self, items):
        """A collection of items represented as a one hot matrix.

        Returns
        -------
        :obj:`torch.autograd.variable.Variable` of :obj:`torch.FloatTensor`
            The items represented as a one hot matrix.
        """
        items_length = len(items)
        tensor = torch.zeros(items_length, self.count())
        indices = self.indices_from_items(items)
        for index, item_index in enumerate(indices):
            tensor[index][item_index] = 1
        return Variable(tensor)
