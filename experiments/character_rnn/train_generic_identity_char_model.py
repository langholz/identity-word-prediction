#!/usr/bin/env python3
import argparse
import os
import random
import math
import sys
from tqdm import tqdm
import torch.nn
from pathlib import Path
from torch.autograd import Variable
sys.path.append("../..")
from core.rnn import CharacterRNN, RNNConfig
from core.corpus import Corpus
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
    parser.add_argument("--model_type", type = str, default = "LSTM", help = "The model type to use: LSTM or GRU")
    parser.add_argument("--corpus_file_path", type = str, default = "generic_identity_char_corpus.pkl", help = "The file path to save/load the corpus constructed from the data")
    parser.add_argument("--vocabulary_file_path", type = str, default = "identity_char_vocabulary.pkl", help = "The file path to save/load the vocabulary constructed from the data")
    parser.add_argument("--train_data_file_path", type = str, default = None, help = "The file path to the training data to process")
    parser.add_argument("--validation_data_file_path", type = str, default = None, help = "The file path to the validation data to process")
    parser.add_argument("--test_data_file_path", type = str, default = None, help = "The file path to the test data to process")
    parser.add_argument("--layer_count", type = int, default = 2, help = "Amount of layers")
    parser.add_argument("--hidden_size", type = int, default = 100, help = "Amount of hidden units per layer")
    parser.add_argument("--embedding_size", type = int, default = 100, help = "The size of the embedding")
    parser.add_argument("--dropout", type = float, default = 0.0, help = "The dropout applied to each layer")
    parser.add_argument("--learning_rate", type = float, default = 0.01, help = "Initial learning rate")
    parser.add_argument("--epochs", type = int, default = 40, help = "Upper limit of iterations to train")
    parser.add_argument("--save_epoch_modulo", type = int, default = 100, help = "Save epoch modulo")
    parser.add_argument("--seed", type = int, default = 4930981, help = "The seed used for reproducibility")
    parser.add_argument("--model_file_path", type = str, default = "generic_identity_char_model.pt", help = "File path used to save the model")
    parser.add_argument("--train_batch_size", type = int, default = 100, help = "The training batch size")
    parser.add_argument("--eval_batch_size", type = int, default = 50, help = "The evaluation batch_size")
    parser.add_argument("--sequence_length", type = int, default = 200, help = "Length of the sequence")
    parser.add_argument("--init_weight_randomly", type = "bool", default = True, help = "Determines whether or not to initialize the models weights randomly")
    parser.add_argument("--tie_weights", type = "bool", default = False, help = "Determines whether to tie the word embedding and softmax weights")
    parser.add_argument("--use_cuda", type = "bool", default = False, help = "Determines whether or not to use CUDA")
    arguments = parser.parse_args()
    return arguments

def file_exists(file_path):
    """Determines whether or not a file exists."""
    path = Path(file_path)
    exists = path.is_file()
    return exists

def validate_data_files_exist(config):
    """Validates that the file paths to the data exist."""
    if (not file_exists(config.train_data_file_path)):
        raise ValueError("Train data file not found: '" + config.train_data_file_path + "'")
    if (not file_exists(config.validation_data_file_path)):
        raise ValueError("Validation data file not found: '" + config.validation_data_file_path + "'")
    if (not file_exists(config.test_data_file_path)):
        raise ValueError("Test data file not found: '" + config.test_data_file_path + "'")

def setup_corpus(config):
    """Constructs and saves, or loads a corpus."""
    if (file_exists(config.corpus_file_path)):
        corpus = Corpus.load(config.corpus_file_path)
        print("Loaded corpus from '" + config.corpus_file_path + "'...")
        if (file_exists(config.vocabulary_file_path)):
            vocabulary = Vocabulary.load(config.vocabulary_file_path)
            corpus.vocabulary = vocabulary
            print("Loaded corpus from '" + config.vocabulary_file_path + "'...")
        else:
            raise ValueError("Vocabulary file not found: '" + config.vocabulary_file_path + "'")
    else:
        print("Processing data and creating corpus...")
        validate_data_files_exist(config)
        corpus = Corpus(config.train_data_file_path, config.validation_data_file_path, config.test_data_file_path)
        vocabulary = None
        if (file_exists(config.vocabulary_file_path)):
            vocabulary = Vocabulary.load(config.vocabulary_file_path)
        identity_processor = IdentityDataProcessor(
            vocabulary = vocabulary,
            eval_batch_size = config.eval_batch_size, 
            sequence_length = config.sequence_length,
            use_cuda = config.use_cuda)
        corpus = identity_processor.process_corpus(corpus)
        vocabulary = corpus.vocabulary
        corpus.vocabulary = None
        corpus.save(config.corpus_file_path)
        print("Saved corpus to '" + config.corpus_file_path + "'...")
        vocabulary.save(config.vocabulary_file_path)
        print("Saved vocabulary to '" + config.vocabulary_file_path + "'...")
        corpus.vocabulary = vocabulary
    
    print("Vocabulary: " + str(corpus.vocabulary.items()))
    return corpus

def save(model_file_path, rnn):
    """Saves the RNN model to a given file path."""
    TorchHelper.save(rnn, model_file_path)
    print("Saved model to '%s'..." % model_file_path)

def load(model_file_path, use_cuda):
    """Loads an RNN model from a given file path."""
    rnn = TorchHelper.load(model_file_path, use_cuda)
    print("Loaded model from '%s'..." % model_file_path)
    return rnn

def create_rnn_config(config, corpus):
    """Creates an RNN configuration from the parsed arguments and a corpus."""
    rnn_config = RNNConfig(
        corpus.vocabulary.count(),
        config.embedding_size,
        config.hidden_size,
        layer_count = config.layer_count,
        type = config.model_type,
        init_weight_randomly = config.init_weight_randomly,
        dropout = config.dropout,
        tie_weights = config.tie_weights)
    return rnn_config

def setup_rnn(config, corpus):
    """Constructs or loads an RNN model."""
    if (file_exists(config.model_file_path)):
        rnn = load(config.model_file_path, config.use_cuda)
    else:
        rnn_config = create_rnn_config(config, corpus)
        rnn = CharacterRNN(rnn_config)
        if (config.use_cuda):
            rnn.cuda()
    return rnn

def repackage_hidden(hidden):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(hidden) == Variable:
        return Variable(hidden.data)
    else:
        return tuple(repackage_hidden(v) for v in hidden)

def evaluate_validation(config, corpus, rnn, criterion):
    """Evaluates the model using the validation data set."""
    loss = evaluate(config, corpus, rnn, criterion, "validation")
    return loss

def evaluate_test(config, corpus, rnn, criterion):
    """Evaluates the model using the test data set."""
    loss = evaluate(config, corpus, rnn, criterion, "test")
    return loss

def evaluate(config, corpus, rnn, criterion, data_set):
    """Evaluates a model given a data set."""
    rnn.eval()
    total_loss = 0
    if ((data_set != "validation") and (data_set != "test")):
        raise ValueError("Invalid data_set value: '" + data_set + "'")
    evaluation_data = corpus.validation if (data_set == "validation") else corpus.test
    hidden = rnn.create_hidden(config.eval_batch_size)
    for index in tqdm(range(len(evaluation_data)), desc = "Evaluating (" + data_set + ")"):
        input_data, target_data = evaluation_data[index]
        for index in range(config.sequence_length):
            output, hidden = rnn(input_data[:, index], hidden)
            total_loss += criterion(output.view(config.eval_batch_size, -1), target_data[:, index])
        hidden = repackage_hidden(hidden)
    return total_loss.data[0] / len(evaluation_data)

def train_epoch(config, corpus, rnn, criterion, optimizer):
    """Trains an epoch."""
    rnn.train()
    total_loss = 0
    hidden = rnn.create_hidden(config.train_batch_size)
    rnn.zero_grad()
    input_data, target_data = corpus.create_random_batch(
        corpus.train,
        config.train_batch_size,
        config.sequence_length,
        use_cuda = config.use_cuda)
    for index in range(config.sequence_length):
        output, hidden = rnn(input_data[:, index], hidden)
        total_loss += criterion(output.view(config.train_batch_size, -1), target_data[:, index])
    total_loss.backward()
    optimizer.step()
    return total_loss.data[0] / config.sequence_length

def train(config, corpus, rnn, criterion, optimizer):
    """Trains the model fully given the configuration."""
    min_validation_loss = None
    avg_loss = 0
    try:
        for epoch in tqdm(range(1, config.epochs + 1), desc = "Training"):
            train_loss = train_epoch(config, corpus, rnn, criterion, optimizer)
            avg_loss += train_loss
            print("\n[epoch = {:3d}] train_loss = {:5.2f}\n".format(epoch, train_loss))
            if ((epoch % config.save_epoch_modulo) == 0):
                validation_loss = evaluate_validation(config, corpus, rnn, criterion)
                print("\n[epoch = {:3d}] validation_loss = {:5.2f}\n".format(epoch, validation_loss))
                if ((min_validation_loss is None) or (min_validation_loss > validation_loss)):
                    min_validation_loss = validation_loss
                    save(config.model_file_path, rnn)
                    print("[epoch = {:3d}] min_validation_loss = {:5.2f}\n".format(epoch, min_validation_loss))
    except KeyboardInterrupt:
        print("\nExiting early...")

def main():
    """The main entry point for training."""
    config = parse_arguments()
    TorchHelper.set_seed(config.seed, use_cuda = config.use_cuda)
    random.seed(config.seed)
    corpus = setup_corpus(config)
    rnn = setup_rnn(config, corpus)
    criterion = TorchHelper.create_cross_entropy_loss()
    optimizer = TorchHelper.create_adam_optimizer(rnn.parameters(), config.learning_rate)
    train(config, corpus, rnn, criterion, optimizer)
    print("Test validation:")
    test_loss = evaluate_test(config, corpus, rnn, criterion)
    print("\ntest_loss = {:5.2f}".format(test_loss))

main()
