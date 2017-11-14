#!/bin/bash
export PYTHONHASHSEED=2157
python -m recognai.run train experiments/definitions/baseline_boe_classifier.json -s experiments/output/boe_simple &
python -m recognai.run train experiments/definitions/baseline_boe_classifier_fasttext_embeddings_tunable.json -s experiments/output/boe_emb_tunable &
python -m recognai.run train experiments/definitions/baseline_boe_classifier_fasttext_embeddings_fixed.json -s experiments/output/boe_emb_fixed &
python -m recognai.run train experiments/definitions/cnn_classifier.json -s experiments/output/cnn_simple &
python -m recognai.run train experiments/definitions/cnn_classifier_fasttext_embeddings_tunable.json -s experiments/output/cnn_emb_tunable &
python -m recognai.run train experiments/definitions/cnn_classifier_fasttext_embeddings_fixed.json -s experiments/output/cnn_emb_fixed &
python -m recognai.run train experiments/definitions/gru_classifier.json -s experiments/output/gru_simple &
python -m recognai.run train experiments/definitions/gru_classifier_fasttext_embeddings_tunable.json -s experiments/output/gru_emb_tunable &
python -m recognai.run train experiments/definitions/gru_classifier_fasttext_embeddings_fixed.json -s experiments/output/gru_emb_fixed &
