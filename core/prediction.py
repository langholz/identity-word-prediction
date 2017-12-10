import torch
from torch.autograd import Variable
from .common_helpers import TorchHelper

class RNNPredictor(object):
    """Defines the `RNNPredictor` class used to predict sequences given an RNN model."""
    def __init__(self, rnn, vocabulary, tensor_from_item):
        """Initializes an instance of the `RNNPredictor` class.

        Parameters
        ----------
        rnn : :obj:`RNN`
            The recurrent neural network model to predict with.
        vocabulary : :obj:`Vocabulary`
            The vocabulary to use to construct a sequence.
        tensor_from_item : :obj:`function`
            A function that converts an item into a tensor.
            When called, the first argument is the vocabulary of type `Vocabulary` and
            the second argument is the actual predicted tensor of type `object`.
        """
        self._rnn = rnn
        self._vocabulary = vocabulary
        self._tensor_from_item = tensor_from_item
        self._rnn.eval()

    def next_state(self, state, input):
        """Updates the state of the model given an input.

        Parameters
        ----------
        state : :obj:`tuple`
            The state of the model.
        input
            The input to update the model with.

        Returns
        -------
        :obj:`tuple`
            The updated state of the model.
        """
        output, hidden = state
        output, hidden = self._rnn(input, hidden)
        state = output, hidden
        return state

    def next_state_given_item(self, state, item):
        """Updates the state of the model given an item.

        Parameters
        ----------
        state : :obj:`tuple`
            The state of the model.
        item : :obj:`object`
            The item to update the model with.

        Returns
        -------
        :obj:`tuple`
            The updated state of the model.
        """
        input = self._tensor_from_item(self._vocabulary, item)
        state = self.next_state(state, input)
        return state

    def create_state(self, context, batch_size = 1):
        """Initializes and creates the state of the model given a context.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        batch_size : :obj:`int`, optional
            The batch size of the models hidden statel By default its one.

        Returns
        -------
        :obj:`tuple`
            The updated state of the model.
        """
        output = None
        hidden = self._rnn.create_hidden(batch_size)
        state = output, hidden
        for item in context:
            state = self.next_state_given_item(state, item)
        return state

    def predict_many_next_indices(self, state, temperatures):
        """Predicts many next item indicesgiven a the models state and multiple temperatures.

        Parameters
        ----------
        state : :obj:`tuple`
            The state of the model.
        temperatures : :obj:`list` of :obj:`float`
            The prediction temperatures.

        Returns
        -------
        :obj:`list` of :obj:`int`
            The predicted indices.
        """
        output, hidden = state
        output_data = output.squeeze().data.cpu()
        indices = []
        for temperature in temperatures:
            output_distribution = output_data.div(temperature).exp()
            index = torch.multinomial(output_distribution, 1)[0]
            indices.append(index)
        return indices

    def predict_next_index(self, state, temperature):
        """Predicts the next index of an item given a the models state and a temperature.

        Parameters
        ----------
        state : :obj:`tuple`
            The state of the model.
        temperature : :obj:`float`
            The prediction temperature.

        Returns
        -------
        :obj:`int`
            The predicted index.
        """
        indices = self.predict_many_next_indices(state, [temperature])
        index = indices[0]
        return index

    def predict_many_next(self, state, temperatures):
        """Predicts the next items given a the models state and multiple temperatures.

        Parameters
        ----------
        state : :obj:`tuple`
            The state of the model.
        temperature : :obj:`list` of :obj:`float`
            The prediction temperatures.

        Returns
        -------
        :obj:`list` of :obj:`object`
            The predicted items.
        """
        indices = self.predict_many_next_indices(state, temperatures)
        items = self._vocabulary.items_from_indices(indices)
        return items

    def predict_next(self, state, temperature):
        """Predicts the next item given a the models state and a temperature.

        Parameters
        ----------
        state : :obj:`tuple`
            The state of the model.
        temperature : :obj:`float`
            The prediction temperature.

        Returns
        -------
        :obj:`object`
            The predicted item.
        """
        index = self.predict_next_index(state, temperature)
        item = self._vocabulary.item_from_index(index)
        return item

    def predict_while_iterator(self, context, temperature, condition):
        """Generates a items given a context and a temperature while the provided condition resolves to True.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperature : :obj:`float`
            The prediction temperature.
        condition : :obj:`function`
            The condition function.

        Notes
        -----
        The condition gets called on every iteration prior to generating the
        value and two parameters are provided in the following order:
            count : :obj:`int`
                The count of the accumulated predicted values.
            next_item : :obj:`object`
                The next predicted item.

        Yields
        ------
        :obj:`object`
            The next predicted item.

        Examples
        --------
        >>> def not_end_of_word(count, item):
        >>>     return item != " " and item != "<eos>"
        >>> print([item for item in predictor.predict_while_iterator(context, temperature, not_end_of_word)])
        """
        state = self.create_state(context)
        count = 0
        while True:
            item = self.predict_next(state, temperature)
            count += 1
            if (condition(count, item)):
                yield item
                state = self.next_state_given_item(state, item)
            else:
                break

    def predict_iterator(self, context, temperature, count):
        """Generates a number of items given a context and a temperature.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperature : :obj:`float`
            The prediction temperature.
        count : :obj:`int`
            The number of items to predict.

        Yields
        ------
        :obj:`object`
            The next predicted item.
        """
        state = self.create_state(context)
        for iteration in range(count):
            item = self.predict_next(state, temperature)
            yield item
            state = self.next_state_given_item(state, item)

    def predict_while(self, context, temperature, condition):
        """Predicts items given a context and a temperature while the provided condition resolves to True.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperature : :obj:`float`
            The prediction temperature.
        condition : :obj:`function`
            The condition function.

        Notes
        -----
        The condition gets called on every iteration prior to generating the
        value and two parameters are provided in the following order:
            count : :obj:`int`
                The count of the accumulated predicted values.
            next_item : :obj:`object`
                The next predicted item.

        Returns
        -------
        :obj:`list` of :obj:`object`
            The list of predicted items.
        """
        items = []
        for item in self.predict_while_iterator(context, temperature, condition):
            items.append(item)
        return items


    def predict(self, context, temperature, count):
        """Predicts a number of items given a context and a temperature.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperature : :obj:`float`
            The prediction temperature.
        count : :obj:`int`
            The number of items to predict.

        Returns
        -------
        :obj:`list` of :obj:`object`
            The list of predicted items.
        """
        items = []
        for item in self.predict_iterator(context, temperature, count):
            items.append(item)
        return items

class CharacterRNNPredictor(RNNPredictor):
    """Defines the `CharacterRNNPredictor` class used to predict characters or words when using a `CharacterRNN` model."""
    def __init__(self, rnn, vocabulary, use_cuda = False):
        """Initializes an instance of the `CharacterRNNPredictor` class.

        Parameters
        ----------
        rnn : :obj:`CharacterRNN`
            The character recurrent neural network model to predict with.
        vocabulary : :obj:`Vocabulary`
            The vocabulary to use to construct a sequence.
        use_cuda : :obj:`bool`, optional
            Determines whether or not to use CUDA.
        """
        super().__init__(rnn, vocabulary, self.tensor_from_item)
        self.word_count = 0
        self.max_word_count = 0
        self._input = Variable(torch.zeros(1, 1).long(), volatile = True)
        if (use_cuda):
            self._input.data = self._input.data.cuda()

    def tensor_from_item(self, vocabulary, item):
        """Converts an item into a tensor.

        Parameters
        ----------
        vocabulary : :obj:`Vocabulary`
            The vocabulary of items.
        item : :obj:`object`
            The item to convert into a tensor.

        Returns
        -------
        :obj:`torch.LongTensor` of size 1x1
            The tensor representing the item.
        """
        index = vocabulary.index_from_item(item)
        self._input.data.fill_(index)
        return self._input

    def _words_condition(self, count, item):
        if (item == " " or item == "<eos>"):
            self.word_count += 1
        return self.word_count <= self.max_word_count

    def _sentence_condition(self, count, item):
        if (item == " " or item == "<eos>"):
            self.word_count += 1
        return (self.word_count <= self.max_word_count) and (item != "<eos>")

    def predict_chars(self, context, temperature, count):
        """Predicts characters given a context and a temperature.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperature : :obj:`float`
            The prediction temperature.
        count : :obj:`int`
            The number of items to predict.

        Returns
        -------
        :obj:`list` of :obj:`object`
            The list of predicted items.
        """
        predictions = super().predict(context, temperature, count)
        return predictions

    def predict_words(self, context, temperature, count):
        """Predicts a words given a context and a temperature.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperature : :obj:`float`
            The prediction temperature.
        count : :obj:`int`
            The number of words to predict.

        Returns
        -------
        :obj:`list` of :obj:`object`
            The list of predicted items.
        """
        self.word_count = 0
        self.max_word_count = count
        predictions = super().predict_while(context, temperature, self._words_condition)
        return predictions

    def predict_sentence(self, context, temperature, max_word_count):
        """Predicts a sentence given a context and a temperature.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperature : :obj:`float`
            The prediction temperature.
        max_word_count : :obj:`int`
            The maximum number of words to predict.

        Returns
        -------
        :obj:`list` of :obj:`object`
            The list of predicted items.
        """
        self.word_count = 0
        self.max_word_count = max_word_count
        predictions = super().predict_while(context, temperature, self._sentence_condition)
        return predictions

    def predict_sentences(self, context, temperatures, max_word_count):
        """Predicts a sentence given a context and multiple temperatures.

        Parameters
        ----------
        context : :obj:`list` of :obj:`object`
            The context to initialize the model with before predicting.
        temperatures : :obj:`float`
            The prediction temperatures.
        max_word_count : :obj:`int`
            The maximum number of words to predict.

        Returns
        -------
        :obj:`list` of :obj:`list` of :obj:`object`
            The list of predicted sentences.
        """
        predictions = []
        for index in range(len(temperatures)):
            temperature = temperatures[index]
            prediction = self.predict_sentence(context, temperature, max_word_count)
            predictions.append(prediction)
        return predictions
