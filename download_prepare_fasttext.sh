#!/bin/bash
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec
mkdir experiments/data/vectors
tar -zcvf experiments/data/vectors/fasttext_es.tar.gz wiki.es.vec
rm wiki.es.vec
