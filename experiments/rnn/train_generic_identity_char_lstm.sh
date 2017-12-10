#!/bin/bash
rm -rf generic_identity_char_lstm.pt
python3 train_generic_identity_char_model.py --epochs 40 --use_cuda True --model_type "LSTM" --model_file_path "generic_identity_char_lstm.pt" --train_data_file_path "../../data/movie dialog corpus/train.txt" --validation_data_file_path "../../data/movie dialog corpus/valid.txt" --test_data_file_path "../../data/movie dialog corpus/test.txt" > generic_identity_char_lstm.txt
