#!/bin/bash
rm -rf e100/ptb_char_lstm.pt
python3 train_char_model.py \
    --epochs 100 \
    --use_cuda True \
    --model_type "LSTM" \
    --model_file_path "e100/ptb_char_lstm.pt" \
    --train_data_file_path "../data/penn/train.txt" \
    --validation_data_file_path "../data/penn/valid.txt" \
    --test_data_file_path "../data/penn/test.txt" > e100/ptb_char_lstm.txt
