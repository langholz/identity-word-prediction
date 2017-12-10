#!/bin/bash
rm -rf e100/ptb_word_gru.pt
python3 train_word_model.py \
    --epochs 100 \
    --use_cuda True \
    --model_type "GRU" \
    --model_file_path "e100/ptb_word_gru.pt" \
    --train_data_file_path "../data/penn/train.txt" \
    --validation_data_file_path "../data/penn/valid.txt" \
    --test_data_file_path "../data/penn/test.txt" > e100/ptb_word_gru.txt
