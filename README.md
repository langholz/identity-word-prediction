# identity-word-prediction
Identity word prediction

## Setup
```
./setup.sh
```

## Run web app locally
Runs the web app locally with the default set of models on port 8899 ([http://0.0.0.0:8899/](http://0.0.0.0:8899/))
```
cd server
./run.sh
```

## Train Generic model
```
python3 train_generic_identity_char_model.py \
    --epochs 20000 \
    --use_cuda True \
    --model_type "LSTM" \
    --embedding_size 1000 \
    --model_file_path "generic_identity_char_lstm.pt" \
    --train_data_file_path "./data/movie dialog corpus/train.txt" \
    --validation_data_file_path "./data/movie dialog corpus/valid.txt" \
    --test_data_file_path "./data/movie dialog corpus/test.txt"
```

## Train Homer Simpson model
(make sure you train the generic model before)
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
    --test_data_file_path "./data/homer simpson/test.txt"
```

## Train Sherlock Holmes model
(make sure you train the generic model before)
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
    --test_data_file_path "./data/sherlock holmes/test.txt"
```

## Documentation
* [Core API](https://langholz.github.io/identity-word-prediction/)
