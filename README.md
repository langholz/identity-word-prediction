# Identity word prediction
## Summary
Have you ever wanted to sound like someone else? How about texting to your family, friends or anyone else while sounding like one of your favorite characters? Well then, this is the project for you! Through this project we present a way in which we can adapt a general neural language model and personalize it to sound like a given personality or character (e.g. Homer Simpson or Sherlock Holmes). We do this by first training a generic nerual language model (NLM) and then extending it by adding extra layers at the end and freezing the generic weights during training.

## Code
### Layout:
*  `identity_data_processor.py`: the data processor used for processing the models so that they can be used by PyTorch.
*  `predict_identity_char_model.py`: contains the logic used to predict the sequence.
*  `push_documentation.sh`: pushes the documentation in the `html` directory into githubs gh pages.
*  `requirements.txt`: the python requriements used for setup.
*  `setup.sh`: the setup bash script.
*  `train_generic_identity_char_model.py`: contains the logic to train the generic identity model.
*  `train_personalized_identity_char_model.py`: contains the logic to train the personalized identity models.
*  `core/`: the common language model functionality and benchmarks used to implement the project.
*  `data/`: the pre-processed data sets used for training and evaluation.
*  `docs/`: the functionality used to create the documentation from the source code.
*  `experiments/`: the character and word level expermients and models conducted during development.
*  `html/`: the generated documentation.
*  `server/`: the directory contianing the web application which relies on the `core`  functionality.

### [Documentation](https://langholz.github.io/identity-word-prediction/)

## Usage
### Setup dependencies
#### Python 3.6
Go to the the Python [website](https://www.python.org/) and follow the instructions to install Python 3.6.x and [pip3](https://docs.python.org/3/installing/index.html).

#### PyTorch
Go to to the PyTorch [website](http://pytorch.org/) and follow the instructions to install using pip. If you want to reproduce exactly with the version that we used while building this project you should install version 0.2.0 which is available for download in the  [previous versions section](http://pytorch.org/previous-versions/).

#### Other
There are other python modules which we have used to build this project. To install these the following command can be executed:
```
./setup.sh
```

### Web app
#### [Try online](https://goo.gl/maEFuL)
#### Run locally (default)
Runs the web app locally with the default pre-trained models on port 8899 ([http://0.0.0.0:8899/](http://0.0.0.0:8899/))
```
cd server
./run.sh
```

### Models
#### Training
In order for training to occur, the pre-processed data must be available. Currently this is placed within the data directory and there is a sub-directory for each data set. Data sets are split into three files:
*  Training data: `train.txt`
*  Validation data: `valid.txt`
*  Test data: `test.txt`

##### Generic character rnn-lstm model
Trains a two layer model using the previously pre-processed [Movie Dialog Corpus](https://www.kaggle.com/Cornell-University/movie-dialog-corpus).
```
python3 train_generic_identity_char_model.py \
    --epochs 20000 \
    --layer_count 2 \
    --use_cuda True \
    --model_type "LSTM" \
    --embedding_size 1000 \
    --corpus_file_path "generic_identity_char_corpus.pkl" \
    --vocabulary_file_path "identity_char_vocabulary.pkl" \
    --model_file_path "generic_identity_char_lstm.pt" \
    --train_data_file_path "./data/movie dialog corpus/train.txt" \
    --validation_data_file_path "./data/movie dialog corpus/valid.txt" \
    --test_data_file_path "./data/movie dialog corpus/test.txt" > generic_training.txt
```
Four assets are generated from this:
*  Corpus: contains the processed data used to train the model (e.g. `generic_identity_char_corpus.pk`).
*  Vocabulary: contains the tokens used to represent the individual units of the model (e.g. `identity_char_vocabulary.pkl`).
*  Model: contains the weights and biases of the rnn-lstm (e.g. `generic_identity_char_lstm.pt`).
*  Log: contains the updated values of the training, validation and test. (e.g. `generic_training.txt`).

##### Homer Simpson character rnn-lstm model
Trains the model by extending the generic model by two more layers and training it with the previously pre-processed [The Simpsons by the Data](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) containing Homer Simpson script lines.
```
python3 train_personalized_identity_char_model.py \
    --epochs 10000 \
    --force_overfitting True \
    --layer_count 2 \
    --use_cuda True \
    --base_model_file_path "generic_identity_char_lstm.pt" \
    --model_file_path "homer_simpson_char_lstm.pt" \
    --corpus_file_path "homer_simpson_char_corpus.pkl" \
    --vocabulary_file_path "identity_char_vocabulary.pkl" \
    --train_data_file_path "./data/homer simpson/train.txt" \
    --validation_data_file_path "./data/homer simpson/valid.txt" \
    --test_data_file_path "./data/homer simpson/test.txt" > homer_training.txt
```
Three assets are generated from this:
*  Corpus: contains the processed data used to train the model (e.g. `homer_simpson_char_corpus.pkl`).
*  Model: contains the weights and biases of the rnn-lstm (e.g. `homer_simpson_char_lstm.pt`).
*  Log: contains the updated values of the training, validation and test. (e.g. `homer_training.txt`).

##### Sherlock Holmes character rnn-lstm model
Trains the model by extending the generic model by tow more layers and training it with the previously pre-processed [Sherlock (BBC TV Series)](https://arianedevere.dreamwidth.org/56441.html) containing Sherlock Holmes script lines.
```
python3 train_personalized_identity_char_model.py \
    --epochs 10000 \
    --force_overfitting True \
    --layer_count 2 \
    --use_cuda True \
    --base_model_file_path "generic_identity_char_lstm.pt" \
    --model_file_path "sherlock_holmes_char_lstm.pt" \
    --corpus_file_path "sherlock_holmes_char_corpus.pkl" \
    --vocabulary_file_path "identity_char_vocabulary.pkl" \
    --train_data_file_path "./data/sherlock holmes/train.txt" \
    --validation_data_file_path "./data/sherlock holmes/valid.txt" \
    --test_data_file_path "./data/sherlock holmes/test.txt" > sherlock_training.txt
```
Three assets are generated from this:
*  Corpus: contains the processed data used to train the model (e.g. `sherlock_holmes_char_corpus.pkl`).
*  Model: contains the weights and biases of the rnn-lstm (e.g. `sherlock_holmes_char_lstm.pt`).
*  Log: contains the updated values of the training, validation and test. (e.g. `sherlock_training.txt`).

#### Prediction
In order to predict, there are two files that are needed that are constructed during training:
*  Vocabulary: The vocabulary for the model. In this case it corresponds to the one created by the generic model which was then reused for the personalized models (e.g. `identity_char_vocabulary.pkl`).
*  Model: The model to use for prediction. (e.g. `homer_simpson_char_lstm.pt`, `sherlock_holmes_char_lstm.pt` or `generic_identity_char_lstm.pt`)

##### Characters
Predicts a defined number of characters `count` (6), given the `context` and a given `temperature`.
```
python3 predict_identity_char_model.py \
    --add_prefix_eos \
    --vocabulary_file_path "identity_char_vocabulary.pkl" \
    --model_file_path "homer_simpson_char_lstm.pt" \
    --temperature 0.8 \
    --context "hel" \
    --count 6
```
##### Words
Predicts a defined number of words `count` (5), given the `context` and a given `temperature`.
```
python3 predict_identity_char_model.py \
    --add_prefix_eos \
    --vocabulary_file_path "identity_char_vocabulary.pkl" \
    --model_file_path "homer_simpson_char_lstm.pt" \
    --temperature 0.8 \
    --context "i thought" \
    --add_postfix_space \
    --words \
    --count 5
```

##### Sentences
Predicts a sentence, given the `context` and a given `temperature`. The `<eos>` sequence is considered the end of the sequence or terminating sentence character.
```
python3 predict_identity_char_model.py \
    --add_prefix_eos \
    --vocabulary_file_path "identity_char_vocabulary.pkl"
    --model_file_path "homer_simpson_char_lstm.pt" \
    --temperature 0.8 \
    --context "i thought" \
    --add_postfix_space \
    --sentence
```
