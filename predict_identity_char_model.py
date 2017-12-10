#!/usr/bin/env python3
import argparse
import os
import random
import torch
from pathlib import Path
from torch.autograd import Variable
from core.prediction import CharacterRNNPredictor
from core.vocabulary import Vocabulary
from core.common_helpers import TorchHelper
from identity_data_processor import IdentityDataProcessor

def bool_from_string(value):
    value = value.lower().strip()
    if (value in ("yes", "true", "t", "y", "1")):
        return True
    elif (value in ("no", "false", "f", "n", "0")):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def parse_arguments():
    """Parses the command line arguments and assigns default values if provided."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", (lambda value: bool_from_string(value)))
    # Common
    parser.add_argument("--context", type = str, help = "The context to use for prediction")
    parser.add_argument("--count", type = int, default = 10, help = "The number of characters/words to predict")
    parser.add_argument("--words", action = "store_true", help = "Determines whether to predict words instead of characters or a sentence")
    parser.add_argument("--sentence", action = "store_true", help = "Determines whether to predict a sentence instead of characters or words")
    parser.add_argument("--add_postfix_eos", action = "store_true", help = "Adds an end of string item to the end of context")
    parser.add_argument("--add_prefix_eos", action = "store_true", help = "Adds an end of string item to the beginning of the context")
    parser.add_argument("--add_postfix_space", action = "store_true", help = "Adds a space at the end of the context")
    parser.add_argument("--temperature", type = float, default = 0.8, help = "The prediction diversity")
    parser.add_argument("--vocabulary_file_path", type = str, default = "identity_char_vocabulary.pkl", help = "The file path to load the vocabulary constructed from the data")
    parser.add_argument("--model_file_path", type = str, default = "identity_char_model.pt", help = "File path used to load the model")
    parser.add_argument("--seed", type = int, default = 4930981, help = "The seed used for reproducibility")
    parser.add_argument("--use_cuda", type = "bool", default = False, help = "Determines whether or not to use CUDA")
    arguments = parser.parse_args()
    return arguments

def validate_temperature(temperature):
    """Validates the provided temperature."""
    if (temperature < 1e-3):
        raise ValueError("Temperature is less than 1e-3")

def file_exists(file_path):
    """Determines whether or not a file exists."""
    path = Path(file_path)
    exists = path.is_file()
    return exists

def setup_vocabulary(config):
    """Loads the supported vocabulary."""
    if (file_exists(config.vocabulary_file_path)):
        vocabulary = Vocabulary.load(config.vocabulary_file_path)
        print("Loaded vocabulary from '" + config.vocabulary_file_path + "'...")
    else:
        raise ValueError("Vocabulary file not found: '" + config.vocabulary_file_path + "'")
    return vocabulary

def load(model_file_path, use_cuda):
    """Loads an RNN model from a given file path."""
    rnn = TorchHelper.load(model_file_path, use_cuda)
    print("Loaded model from '%s'..." % model_file_path)
    return rnn

def setup_rnn(config):
    """Loads an existing RNN model."""
    if (file_exists(config.model_file_path)):
        rnn = load(config.model_file_path, config.use_cuda)
    else:
        raise ValueError("Model file not found: '" + config.model_file_path + "'")
    return rnn

def setup_rnn_predictor(rnn, vocabulary):
    """Creates a predictor given the model and its vocabulary."""
    rnn_predictor = CharacterRNNPredictor(rnn, vocabulary)
    return rnn_predictor

def setup_context(config):
    """Configures by normalizing the provided context."""
    identity_processor = IdentityDataProcessor()
    context = identity_processor.items_from_string(config.context)
    if (config.add_prefix_eos):
        context = ["<eos>"] + context
    if (not config.add_postfix_eos):
        context = context[:-1]
    if (config.add_postfix_space):
        context += [" "]
    return context

def predict_items(config, rnn_predictor, context):
    if (config.sentence):
        predictions = rnn_predictor.predict_sentence(context, config.temperature, config.count)
    elif (config.words):
        predictions = rnn_predictor.predict_words(context, config.temperature, config.count)
    else:
        predictions = rnn_predictor.predict_chars(context, config.temperature, config.count)
    print("prediction: \"" + "".join(predictions) + "\"")

def predict(config, rnn_predictor, context):
    """Performs the prediction given the context."""
    print("context: ")
    print("   string: " + config.context)
    print("   items: " + str(context))
    predict_items(config, rnn_predictor, context)

def main():
    """The main entry point for prediction."""
    config = parse_arguments()
    validate_temperature(config.temperature)
    TorchHelper.set_seed(config.seed)
    vocabulary = setup_vocabulary(config)
    rnn = setup_rnn(config)
    if config.use_cuda:
        rnn.cuda()
    else:
        rnn.cpu()

    rnn_predictor = setup_rnn_predictor(rnn, vocabulary)
    context = setup_context(config)
    predict(config, rnn_predictor, context)

if __name__ == "__main__":
    main()
