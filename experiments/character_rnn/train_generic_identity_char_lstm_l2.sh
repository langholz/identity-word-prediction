#!/bin/bash
rm -rf embedding_size_1000/generic_identity_char_lstm_l2.pt
python3 train_generic_identity_char_model.py --epochs 20000 --use_cuda True --model_type "LSTM" --embedding_size 1000 --model_file_path "embedding_size_1000/generic_identity_char_lstm_l2.pt" --train_data_file_path "../../data/movie dialog corpus/train.txt" --validation_data_file_path "../../data/movie dialog corpus/valid.txt" --test_data_file_path "../../data/movie dialog corpus/test.txt" > embedding_size_1000/generic_identity_char_lstm_l2.txt
