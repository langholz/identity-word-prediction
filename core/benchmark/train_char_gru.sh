#!/bin/bash
rm -rf e100/ptb_char_gru.pt
python3 train_char_model.py \
    --epochs 100 \
    --use_cuda True \
    --model_type "GRU" \
    --model_file_path "e100/ptb_char_gru.pt" \
    --train_data_file_path "../data/penn/train.txt" \
    --validation_data_file_path "../data/penn/valid.txt" \
    --test_data_file_path "../data/penn/test.txt" > e100/ptb_char_gru.txt
