#!/bin/bash
rm -rf overfit_es1k_shared_vocab/homer_simpson_char_lstm_e10k_l3_es1k.pt
python3 train_personalized_identity_char_model.py \
    --epochs 10000 \
    --force_overfitting True \
    --use_cuda True \
    --base_model_file_path "embedding_size_1000/generic_identity_char_lstm_l2.pt" \
    --model_file_path "overfit_es1k_shared_vocab/homer_simpson_char_lstm_e10k_l3_es1k.pt" \
    --corpus_file_path "overfit_es1k_shared_vocab/homer_simpson_char_corpus.pkl" \
    --vocabulary_file_path "identity_char_vocabulary.pkl" \
    --train_data_file_path "../../data/homer simpson/train.txt" \
    --validation_data_file_path "../../data/homer simpson/valid.txt" \
    --test_data_file_path "../../data/homer simpson/test.txt" > overfit_es1k_shared_vocab/homer_simpson_char_lstm_e10k_l3_es1k.txt

rm -rf overfit_es1k_shared_vocab/homer_simpson_char_lstm_e10k_l4_es1k.pt
python3 train_personalized_identity_char_model.py \
    --epochs 10000 \
    --force_overfitting True \
    --layer_count 2 \
    --use_cuda True \
    --base_model_file_path "embedding_size_1000/generic_identity_char_lstm_l3.pt" \
    --model_file_path "overfit_es1k_shared_vocab/homer_simpson_char_lstm_e10k_l4_es1k.pt" \
    --corpus_file_path "overfit_es1k_shared_vocab/homer_simpson_char_corpus.pkl" \
    --vocabulary_file_path "identity_char_vocabulary.pkl" \
    --train_data_file_path "../../data/homer simpson/train.txt" \
    --validation_data_file_path "../../data/homer simpson/valid.txt" \
    --test_data_file_path "../../data/homer simpson/test.txt" > overfit_es1k_shared_vocab/homer_simpson_char_lstm_e10k_l4_es1k.txt
