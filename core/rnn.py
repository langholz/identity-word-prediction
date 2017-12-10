import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNConfig(object):
    """Defines a recurrent neural network configuration.

    Attributes
    ----------
    term_count : :obj:`int`
        The amount of terms or items in the vocabulary.
    embedding_size : :obj:`int`
        The size of the embedding.
    hidden_size : :obj:`int`
        The amount of hidden units per layer.
    layer_count : :obj:`int`, optional
        The amount of layers. By default, we create two layers.
    type : :obj:`str`, optional
        The type of recurrent neural network to create: 'LSTM' or 'GRU'.
        By default, we use GRU.
    init_weight_randomly : :obj:`bool`, optional
        Determines whether or not the weights should be initialized randomly.
        By default, we do not initialize the weights randomly.
    dropout : :obj:`float`, optional
        The dropout parameter to use. By default, the value is zero (no dropout).
    tie_weights : :obj:`bool`, optional
        Determines whether or not the weights should be tied.
        By default, weights are not tied.
    """
    def __init__(
        self,
        term_count,
        embedding_size,
        hidden_size,
        layer_count = 2,
        type = "GRU",
        init_weight_randomly = False,
        dropout = 0.0,
        tie_weights = False):
        """Initializes an instance of the `RNNConfig` class.

        Parameters
        ----------
        term_count : :obj:`int`
            The amount of terms or items in the vocabulary.
        embedding_size : :obj:`int`
            The size of the embedding.
        hidden_size : :obj:`int`
            The amount of hidden units per layer.
        layer_count : :obj:`int`
            The amount of layers.
        type : :obj:`str`
            The type of recurrent neural network to create: 'LSTM' or 'GRU'.
        init_weight_randomly : :obj:`bool`
            Determines whether or not the weights should be initialized randomly.
        dropout : :obj:`float`
            The dropout parameter to use.
        tie_weights : :obj:`bool`
            Determines whether or not the weights should be tied.
        """
        self.term_count = term_count
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_count = layer_count
        self.type = type
        self.init_weight_randomly = init_weight_randomly
        self.dropout = dropout
        self.tie_weights = tie_weights

    @classmethod
    def from_serialized(cls, serialized):
        """Creates an instance of the `RNNConfig` class from a dictionary.

        Parameters
        ----------
        serialized : :obj:`dict`
            The dictionary containing the corresponding configuration.

        Returns
        -------
        :obj:`RNNConfig`
            The configuration instance.
        """
        value = cls(
            serialized["term_count"],
            serialized["embedding_size"],
            serialized["hidden_size"],
            serialized["layer_count"],
            serialized["type"],
            serialized["init_weight_randomly"],
            serialized["dropout"],
            serialized["tie_weights"])
        return value

    def _validate_larger_or_equal_to_one(self, name, value):
        if (value < 1.0):
            raise ValueError(str(name) + " (" + str(value) + ") is less than one")

    def _validate_larger_or_equal_to_zero(self, name, value):
        if (value < 0.0):
            raise ValueError(str(name) + " (" + str(value) + ") is less than zero")

    def _validate_type(self):
        if not (self.type in ["LSTM", "GRU"]):
            raise ValueError("An invalid RNN type ('" + self.type + "')")

    def _validate_tie_weights(self):
        if (self.tie_weights):
            if (self.embedding_size != self.hidden_size):
                error = "Tie weights option expects that embedding_size (" + str(self.embedding_size)
                error += ") equals hidden_size (" + str(self.hidden_size) + ")"
                raise ValueError(error)

    def _validate(self):
        self._validate_larger_or_equal_to_one("term_count", self.term_count)
        self._validate_larger_or_equal_to_one("embedding_size", self.embedding_size)
        self._validate_larger_or_equal_to_one("hidden_size", self.hidden_size)
        self._validate_larger_or_equal_to_one("layer_count", self.layer_count)
        self._validate_type()
        self._validate_larger_or_equal_to_zero("dropout", self.dropout)
        self._validate_tie_weights()

    def copy(self):
        """Makes a copy of the instance.

        Returns
        -------
        :obj:`RNNConfig`
            The copy of the configuration.
        """
        serialized = self.serialize()
        rnn_config = RNNConfig.from_serialized(serialized)
        return rnn_config

    def serialize(self):
        """Serializes the instance as a dictionary.

        Returns
        -------
        :obj:`dict`
            The configuration as a dictionary.
        """
        value = {
            "term_count": self.term_count,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "layer_count": self.layer_count,
            "type": self.type,
            "init_weight_randomly": self.init_weight_randomly,
            "dropout": self.dropout,
            "tie_weights": self.tie_weights
        }
        return value

class RNN(nn.Module):
    """Defines a recurrent neural network.

    Attributes
    ----------
    config : :obj:`RNNConfig`
        The recurrent neural network configuration.
    dropout : :obj:`torch.nn.Dropout`
        The dropout to use while training.
    encoder: :obj:`torch.nn.Embedding`
        The item encoder.
    rnn : :obj:`torch.nn.Module`
        The recurrent nerual network.
    decoder : :obj:`torch.nn.Linear`
        The item decoder.
    """
    def __init__(self, config):
        """Initializes an instance of the `RNN` class.

        Parameters
        ----------
        config : :obj:`RNNConfig`
            The configuration to use when initializing the instance.
        """
        super(RNN, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.Embedding(config.term_count, config.embedding_size)
        self.rnn = getattr(nn, config.type)(
            config.embedding_size,
            config.hidden_size,
            config.layer_count,
            dropout = config.dropout)
        self.decoder = nn.Linear(config.hidden_size, config.term_count)
        if (config.tie_weights):
            self.decoder.weight = self.encoder.weight
        if (config.init_weight_randomly):
            self._init_weight()

    def _init_weight(self):
        range_value = 0.1
        self.encoder.weight.data.uniform_(-range_value, range_value)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-range_value, range_value)

    def _create_hidden_variable(self, weight, batch_size):
        variable = Variable(weight.new(self.config.layer_count, batch_size, self.config.hidden_size).zero_())
        return variable

    def forward(self, input, hidden):
        """Defines the computation performed at every call.

        Parameters
        ----------
        input : :obj:``
            The input.
        hidden : :obj:`torch.autograd.Variable`
            The hidden state.

        Returns
        -------
        :obj:`tuple` 
            The first value is the output and the second is the hidden state. 
        """
        encoded_input = self.dropout(self.encoder(input))
        encoded_output, hidden = self.rnn(encoded_input, hidden)
        encoded_output = self.dropout(encoded_output)
        decoder_input = encoded_output.view(encoded_output.size(0) * encoded_output.size(1), encoded_output.size(2))
        decoded_output = self.decoder(decoder_input)
        return decoded_output.view(encoded_output.size(0), encoded_output.size(1), decoded_output.size(1)), hidden

    def create_hidden(self, batch_size = 1):
        """Creates the recurrent neural network hidden state.

        Parameters
        ----------
        batch_size : :obj:`int`, optional
            The input batch size.

        Returns
        -------
        :obj:`torch.autograd.Variable`
            The hidden state.
        """
        hidden = None
        weight = next(self.parameters()).data
        if (self.config.type == "LSTM"):
            hidden = (self._create_hidden_variable(weight, batch_size), self._create_hidden_variable(weight, batch_size))
        elif (self.config.type == "GRU"):
            hidden = self._create_hidden_variable(weight, batch_size)
        else:
            raise ValueError("RNN type should be LSTM or GRU!")
        return hidden

class CharacterRNN(RNN):
    """Defines a character recurrent neural network.

    Attributes
    ----------
    config : :obj:`RNNConfig`
        The recurrent neural network configuration.
    dropout : :obj:`torch.nn.Dropout`
        The dropout to use while training.
    encoder: :obj:`torch.nn.Embedding`
        The item encoder.
    rnn : :obj:`torch.nn.Module`
        The recurrent nerual network.
    decoder : :obj:`torch.nn.Linear`
        The item decoder.
    """
    def __init__(self, config):
        super(CharacterRNN, self).__init__(config)

    def forward(self, input, hidden):
        """Defines the computation performed at every call.

        Parameters
        ----------
        input : :obj:``
            The input.
        hidden : :obj:`torch.autograd.Variable`
            The hidden state.

        Returns
        -------
        :obj:`tuple` 
            The first value is the output and the second is the hidden state. 
        """
        batch_size = input.size(0)
        encoded_input = self.dropout(self.encoder(input))
        encoder_input = encoded_input.view(1, batch_size, -1)
        encoded_output, hidden = self.rnn(encoder_input, hidden)
        encoded_output = self.dropout(encoded_output)
        decoder_input = encoded_output.view(batch_size, -1)
        decoded_output = self.decoder(decoder_input)
        return decoded_output, hidden

class PersonalizedCharacterRNN(CharacterRNN):
    """Defines a personalized character recurrent neural network.

    Attributes
    ----------
    config : :obj:`RNNConfig`
        The recurrent neural network configuration.
    dropout : :obj:`torch.nn.Dropout`
        The dropout to use while training.
    encoder: :obj:`torch.nn.Embedding`
        The item encoder.
    rnn : :obj:`torch.nn.Module`
        The recurrent nerual network.
    decoder : :obj:`torch.nn.Linear`
        The item decoder.
    """
    def __init__(self, config):
        """Initializes an instance of the `RNN` class.

        Parameters
        ----------
        config : :obj:`RNNConfig`
            The configuration to use when initializing the instance.
        """
        super(PersonalizedCharacterRNN, self).__init__(config)

    @classmethod
    def transfer_from_rnn_with_extra_layers(cls, rnn, layer_count, use_cuda = False, fail_on_no_update = True):
        """Creates a personalized RNN by transfering weights from an existing RNN and adding extra layers.

        Parameters
        ----------
        rnn : :obj:`RNN`
            The RNN to transfer the weights from.
        layer_count : :obj:`int`
            The number of extra layers to add.
        use_cuda : :obj:`bool`, optional
            Determines whether or not to use CUDA. By default, False.
        fail_on_no_update : :obj:`bool`, optional
            Determines whether or not to fail if no update. By default, True.

        Returns
        -------
        :obj:`PersonalizedCharacterRNN`
            The personalized recurrent neural network.
        """
        config = rnn.config.copy()
        config.layer_count += layer_count
        personalized_rnn = cls(config)
        if (use_cuda):
            personalized_rnn.cuda()
        updated_keys = []
        forzen_keys = []
        pretrained_state_dict = rnn.state_dict()
        model_state_dict = personalized_rnn.state_dict()
        updated_model_state_dict = {}
        for key, value in model_state_dict.items():
            updated_model_state_dict[key] = value
            if (key in pretrained_state_dict):
                pretrained_value = pretrained_state_dict[key]
                if ((type(value) == type(pretrained_value)) and (value.size() == pretrained_value.size())):
                    updated_model_state_dict[key] = pretrained_value
                    updated_keys.append(key)
                    if (key.startswith("rnn.")):
                        forzen_keys.append(key)
                        updated_model_state_dict[key].requires_grad = False

        if ((fail_on_no_update) and ((len(updated_keys) == 0) or (len(forzen_keys) == 0))):
            raise ValueError("Did not transfer any values from existing RNN")
        else:
            print("Created PersonalizedCharacterRNN with %d extra layers:" % layer_count)
            print("\tTransfered weights for the following keys: " + ", ".join(updated_keys))
            print("\tFroze the following weights: " + ", ".join(forzen_keys))

        model_state_dict.update(updated_model_state_dict)
        personalized_rnn.load_state_dict(updated_model_state_dict)
        return personalized_rnn
