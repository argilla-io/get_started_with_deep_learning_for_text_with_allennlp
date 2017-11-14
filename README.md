# Introduction

This repository contains code and experiments using PyTorch, AllenNLP and spaCy and is intended as a learning resource for getting started with this libraries and with deep learning for NLP technologies.

In particular, it contains:

1. Custom modules for defining a SequenceClassifier and its Predictor.
2. A basic custom DataReader for reading CSV files.
3. An experiments folder containing several experiment JSON files to show how to define a baseline and refine it with more sophisticated approaches.

The overall goal is to classify tweets in Spanish corresponding to the COSET challenge dataset: a collection of tweets for a recent Spanish Election. The winning approach of the challenge is described in the following paper: http://ceur-ws.org/Vol-1881/COSET_paper_7.pdf.

# Setup

Use a virtual environment, Conda for example:

``conda create -n allennlp_spacy``

``source activate allennlp_spacy``

Install PyTorch for your platform:
``pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl``

Install spaCy Spanish model:
``python -m spacy download es``

Install AllenNLP and other dependencies:
``pip install -r requirements.txt``

Install custom module for running AllenNLP commands with custom models:
``python setup.py develop``

Install Tensorboard:
``pip install tensorboard``


Download pre-trained  and prepare word vectors from fastText project:
``download_prepare_fasttext.sh``

# Goals

1. Understand the basic components of AllenNLP and PyTorch.

2. Understand how to configure AllenNLP to use spaCy models in different languages, in this case Spanish model.

3. Understand how to create and connect custom models using AllenNLP and extending its command-line.

4. Design and compare several experiments on a simple Tweet classification tasks in Spanish. Start by defining a simple baseline and progressively use more complex models.

5. Use Tensorboard for monitoring the experiments.

6. Compare your results with existing literature (i.e., results of the COSET Tweet classification challenge)

7. Learn how to prepare and use external pre-trained word embeddings, in this case fastText's wikipedia-based word vectors.

# Exercises

## Inspecting Seq2VecEncoders and understanding the basic building blocks of AllenNLP:

Check the basic structure of these modules in AllenNLP.

## Defining and running our baseline:

In the folder ``experiments/definitions/`` you can find the definition of our baseline, using a BagOfEmbeddingsEncoder.

Run the experiment using:
```shell
python -m recognai.run train experiments/definitions/baseline_boe_classifier.json -s experiments/output/baseline
```

## Monitor your experiments using Tensorboard:

You can monitor your experiments by running TensorBoard and pointing it to the experiments output folder:

```shell
tensorboard --logdir=experiments/output
```

## Defining and running a CNN classifier:

In the folder ``experiments/definitions/`` you can find the definition of a CNN classifier. As you see, we only need to configure a new encoder using a CNN.

Run the experiment using:

```shell
python -m recognai.run train experiments/definitions/cnn_classifier.json -s experiments/output/cnn
```

## Using pre-trained word embeddings:

Facebook fastText's team has made available pre-trained word embeddings for 294 languages (see https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). Using the ``download_prepare_fasttext.sh`` script, you can download the Spanish vectors and use them as pre-trained weights in either of the models.

To use pre-trained embeddings, you can run the experiment using:
```shell
python -m recognai.run train experiments/definitions/cnn_classifier_fasttext_embeddings_fixed.json -s experiments/output/cnn_embeddings_fixed
```

Or use pre-trained embeddings and let the network tune their weights, using:
```shell
python -m recognai.run train experiments/definitions/cnn_classifier_fasttext_embeddings_tunable.json -s experiments/output/cnn_embeddings_tuned
```

## Extra:

- Check https://github.com/recognai/custom_models_allennlp/tree/master/experiments/tweet-classification-spanish and run an RNN classifier. How are the results? Tip: Initialization is key when training LSTMs.

- The network quickly overfits, what strategies would you follow?
