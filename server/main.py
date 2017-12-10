#!/bin/bash python3
import os
import sys
import urllib.parse
import random
import pickle
import torch
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from torch.autograd import Variable
sys.path.append("..")
from core.prediction import CharacterRNNPredictor
from core.vocabulary import Vocabulary
from core.common_helpers import TorchHelper
from identity_data_processor import IdentityDataProcessor

app = Flask(__name__)
max_word_count = 20
seed = 4930981
use_cuda = False
vocabulary_file_path = "./models/identity_char_vocabulary.pkl"
char_rnn_homer_file_path = "./models/homer_simpson_char_lstm_e10k_l4_es1k.pt"
char_rnn_sherlock_file_path = "./models/sherlock_holmes_char_lstm_e10k_l4_es1k.pt"
details_file_path = "./models/analysis/details.pkl"
identity_processor = None
homer_predictor = None
sherlock_predictor = None
details = None

def file_exists(file_path):
    """Determines whether or not a file exists."""
    path = Path(file_path)
    exists = path.is_file()
    return exists

def load(file_path):
    """Loads pickled data."""
    with open(file_path, "rb") as input_handle:
        value = pickle.load(input_handle)
    return value

def raise_if_file_not_exists(file_path):
    """Raises an error if the file does not exist."""
    if (not file_exists(file_path)):
        raise ValueError("File '" + file_path + "'' not found!")

def load_model(model_file_path, use_cuda):
    """Loads an RNN model from a given file path."""
    rnn = TorchHelper.load(model_file_path, use_cuda)
    print("Loaded model from '%s'..." % model_file_path)
    return rnn

def load_vocabulary(file_path):
    """Loads the supported vocabulary."""
    vocabulary = Vocabulary.load(file_path)
    print("Loaded vocabulary from '%s'" % file_path)
    return vocabulary

def normalize_identity(identity):
    """Normalizes the identity."""
    identity = identity.strip().lower()
    return identity

def valid_identity(identity):
    """Determines whether or not the provided identity is a valid value."""
    valid = (identity == "homer") or (identity == "sherlock")
    return valid

def normalize_context(context):
    """Normalizes the context."""
    context = context.strip()
    context_items = identity_processor.items_from_string(context)
    if (len(context_items) > 0):
        context_items[0] = context_items[0].upper()
        context_items = ["<eos>"] + context_items[:-1] + [" "]
    else:
        context_items = ["<eos>"] 
    return context_items

def is_float(value):
    """Determines whether or not the provided value is a float or not."""
    try:
        float(value)
        return True
    except:
        return False

def normalize_temperatures(temperatures):
    """Normalizes a comma-separated string of temperatures into an array of floats."""
    values = temperatures.split(",")
    temperature_values = []
    for value in values:
        value = value.strip()
        if (is_float(value)):
            temperature = float(value)
            temperature_values.append(temperature)
    return temperature_values

def valid_temperature(temperature):
    """Determines whether or not the temperature is valid."""
    return temperature > 1e-3 and temperature <= 1.0

def valid_tempratures(temperatures):
    """Determines whether the temperatures are valid or not."""
    valid = True
    invalid_temperatures = []
    for temperature in temperatures:
        if (not valid_temperature(temperature)):
            valid = False
            invalid_temperatures.append(temperature)
    return valid, invalid_temperatures

def parse_identity(request):
    """Parses the character identity from an http request."""
    success = False
    result = None
    identity = request.args.get("identity")
    if (identity is not None):
        identity = normalize_identity(identity)
        success = valid_identity(identity)
        result = identity if (success) else "Invalid identity: " + request.args.get("identity")
    else:
        success = False
        result = "Identity not provided!"
    return success, result

def parse_context(request):
    """Parses the context from an http request."""
    success = False
    result = None
    context_string = request.args.get("context")
    if (context_string is not None):
        context = normalize_context(context_string)
        success = True
        result = context
    else:
        success = False
        result = "Context not provided!"
    return success, result

def parse_temperatures(request):
    """Parses the temperatures from an http request."""
    success = False
    result = None
    temperatures_string = request.args.get("temperatures")
    if (temperatures_string is not None):
        temperatures = normalize_temperatures(temperatures_string)
        success = len(temperatures) > 0
        if (success):
            success, invalid_temperatures = valid_tempratures(temperatures)
            result = temperatures if (success) else ("Invalid temperatures provided: " + str(invalid_temperatures))
        else:
            result = "Temperatures not provided!"
    else:
        success = False
        result = "Temperatures not provided!"
    return success, result

def parse_sentence_prediction_request(request):
    """Parses the sentences prediction http request."""
    identity = None
    context = None
    temperatures = None
    success, result = parse_identity(request)
    if (success):
        identity = result
        success, result = parse_context(request)
    if (success):
        context = result
        success, result = parse_temperatures(request)
    if (success):
        temperatures = result
        result = (identity, context, temperatures)
    return success, result

def predict_sentences(identity, context, temperatures):
    """Predicts a set of sentences given an identity, context and temperatures."""
    predictions = None
    TorchHelper.set_seed(seed)
    if (identity == "homer"):
        predictions = homer_predictor.predict_sentences(context, temperatures, max_word_count)
    elif (identity == "sherlock"):
        predictions = sherlock_predictor.predict_sentences(context, temperatures, max_word_count)
    return predictions

def sentences_prediction(identity, context, temperatures):
    """Consolidates sentence prediction."""
    result = { "sentences": [] }
    predictions = predict_sentences(identity, context, temperatures)
    if (predictions is not None):
        for index in range(len(temperatures)):
            temperature = temperatures[index]
            characters = predictions[index]
            text = "".join(characters)
            words = text.split(" ")
            if (len(words) > 0 and text != ""):
                sentence_info = { "index": index, "temperature": temperature, "words": words, "text": text }
                result["sentences"].append(sentence_info)
    return result

@app.route("/")
def index():
    """The index http entry point route."""
    return render_template("index.html")

@app.route("/prediction", methods = ["GET"])
def prediction():
    """The prediction http entry point route."""
    code = 400
    success, result = parse_sentence_prediction_request(request)
    if (success):
        prediction = sentences_prediction(*result)
        if (len(prediction["sentences"]) > 0):
            identity, context, temperatures = result
            context_string = "".join(context[1:])
            result = render_template(
                "prediction.html",
                identity = identity,
                context = context_string,
                sentences = prediction["sentences"])
            code = 200
        else:
            result = "Unable to predict sentences!"
            code = 500
    else:
        code = 400
    return result, code

@app.route("/details", methods = ["GET"])
def details():
    """The main index http entry point route."""
    code = 200
    result = render_template("details.html", details = details)
    return result, code

@app.route("/about", methods = ["GET"])
def about():
    """The about http entry point route."""
    code = 200
    result = render_template("about.html")
    return result, code

@app.route("/predict/sentences", methods = ["GET"])
def predict_sentences_json():
    """The sentence prediction rest api entry point route."""
    code = 400
    success, result = parse_sentence_prediction_request(request)
    if (success):
        result = sentences_prediction(*result)
        code = 200
        result = jsonify(result)
    else:
        code = 400
    return result, code

def setup(vocabulary_file_path, homer_model_file_path, sherlock_model_file_path, details_file_path, use_cuda):
    """Setup the models and settings for use within flask."""
    raise_if_file_not_exists(vocabulary_file_path)
    raise_if_file_not_exists(homer_model_file_path)
    raise_if_file_not_exists(sherlock_model_file_path)
    raise_if_file_not_exists(details_file_path)
    identity_processor = IdentityDataProcessor()
    vocabulary = load_vocabulary(vocabulary_file_path)
    char_rnn_homer = load_model(homer_model_file_path, use_cuda)
    char_rnn_sherlock = load_model(sherlock_model_file_path, use_cuda)
    homer_predictor = CharacterRNNPredictor(char_rnn_homer, vocabulary)
    sherlock_predictor = CharacterRNNPredictor(char_rnn_sherlock, vocabulary)
    details = load(details_file_path)
    return identity_processor, homer_predictor, sherlock_predictor, details

if __name__ == "__main__":
    port = int(sys.argv[1])
    identity_processor, homer_predictor, sherlock_predictor, details = setup(
        vocabulary_file_path,
        char_rnn_homer_file_path,
        char_rnn_sherlock_file_path,
        details_file_path,
        use_cuda)
    app.run(host = "0.0.0.0", port = port)
