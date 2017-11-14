# Introduction

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




## Using pre-trained word embeddings:

Facebook fastText's team has made available pre-trained word embeddings for 294 languages (see https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). Using the ``download_prepare_fasttext.sh`` script, you can download the Spanish vectors and use them as pre-trained weights in either of the models.
