#!/bin/bash
rm -rf generic_identity_char_gru_dropout.pt
python3 train_generic_identity_char_model.py --epochs 40 --dropout 0.2 --use_cuda True --model_type "GRU" --model_file_path "generic_identity_char_gru_dropout.pt" --train_data_file_path "../../data/movie dialog corpus/train.txt" --validation_data_file_path "../../data/movie dialog corpus/valid.txt" --test_data_file_path "../../data/movie dialog corpus/test.txt" > generic_identity_char_gru_dropout.txt

