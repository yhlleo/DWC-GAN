import os
import pickle
import numpy as np
from collections import OrderedDict

# requiring install fastText
from fasttext import load_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fasttext_model', type=str, default='datasets/wiki-embeddings/wiki.en.bin', 
    help='pretrained fastText model (binary file)')
parser.add_argument('--output_dict', type=str, default='embeddings.npy')
opts = parser.parse_args()

# provide a vocabulary list
words = [
    "this",
    "is",
    "an",
    "example"
]

# please download pretrained word embeddings from the web page: https://fasttext.cc/docs/en/pretrained-vectors.html
# or directly download from: https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
# load pretrained word embeddings
embeddings = load_model(opts.fasttext_model)

words_dict = OrderedDict()
for wd in words:
    words_dict[wd] = embeddings.get_word_vector(wd)

# save embeddings
with open(opts.output_dict, 'rb') as fin:
    words_dict = pickle.load(fin)
