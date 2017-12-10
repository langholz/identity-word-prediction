#!/usr/bin/env python3
import argparse
import os
import random
import torch
import sys
from pathlib import Path
from torch.autograd import Variable
sys.path.append("../..")
from core.prediction import RNNPredictor
from core.corpus import Corpus
from core.data_processor import PennTreeBankProcessor
from core.common_helpers import TorchHelper

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
    parser.add_argument("--count", type = int, help = "The number of items to predict")
    parser.add_argument("--add_postfix_eos", action = "store_true", help = "Adds an end of string item to the end of context")
    parser.add_argument("--add_prefix_eos", action = "store_true", help = "Adds an end of string item to the beginning of the context")
    parser.add_argument("--add_postfix_space", action = "store_true", help = "Adds a space at the end of the context")
    parser.add_argument("--temperature", type = float, default = 1.0, help = "The prediction diversity")
    parser.add_argument("--corpus_file_path", type = str, default = "ptb_char_corpus.pkl", help = "The file path to load the corpus constructed from the data")
    parser.add_argument("--model_file_path", type = str, default = "ptb_char_model.pt", help = "File path used to load the model")
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

def setup_corpus(config):
    """Loads a an existing corpus."""
    if (file_exists(config.corpus_file_path)):
        corpus = Corpus.load(config.corpus_file_path)
        print("Loaded corpus from '" + config.corpus_file_path + "'...")
    else:
        raise ValueError("Corpus file not found: '" + config.corpus_file_path + "'")
    return corpus

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

input = Variable(torch.zeros(1, 1).long(), volatile = True)

def setup_input(config):
    if (config.use_cuda):
        input.data = input.data.cuda()

def tensor_from_item(vocabulary, item):
    """Updates the input tensor given an item."""
    index = vocabulary.index_from_item(item)
    input.data.fill_(index)
    return input

def setup_rnn_predictor(rnn, corpus):
    """Creates a predictor given the model and its corpus."""
    rnn_predictor = RNNPredictor(rnn, corpus.vocabulary, tensor_from_item)
    return rnn_predictor

def setup_context(config):
    """Configures by normalizing the provided context."""
    ptb_processor = PennTreeBankProcessor()
    context = ptb_processor.character_items_from_string(config.context)
    context = context[:-1]
    if (config.add_prefix_eos):
        context = ["<eos>"] + context
    if (not config.add_postfix_eos):
        context = context[:-1]
    if (config.add_postfix_space):
        context += [" "]
    return context

def predict(config, rnn_predictor, context):
    """Performs the prediction given the context."""
    print("context: ")
    print(config.context)
    print(context)
    predictions = rnn_predictor.predict(context, config.temperature, config.count)
    print("".join(predictions))

def main():
    """The main entry point for prediction."""
    config = parse_arguments()
    validate_temperature(config.temperature)
    setup_input(config)
    TorchHelper.set_seed(config.seed)
    corpus = setup_corpus(config)
    rnn = setup_rnn(config)
    if config.use_cuda:
        rnn.cuda()
    else:
        rnn.cpu()

    rnn_predictor = setup_rnn_predictor(rnn, corpus)
    context = setup_context(config)
    predict(config, rnn_predictor, context)

main()
