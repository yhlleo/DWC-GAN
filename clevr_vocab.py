import torch
import random

import numpy as np
PAD, BOS, EOS, UNK = '<_>', '<bos>', '<eos>', '<unk>'

vocab_clevr_list = [ # Clevr words
    "cylinder",
    "cube",
    "sphere",

    "brown",
    "gray",
    "green",
    "purple",
    "blue",
    "yellow",
    "cyan",
    "red",

    "large",
    "small",

    "rubber",
    "metal",

    "move",
    "change",
    "add",
    "remove",
    "shift",
    "transfer",
    "alter",
    "modify",
    "append",
    "delete",
    "erase",
    
    "to",
    "of",
    "right",
    "left",
    "a",
    ".",
    ",",
    " ",
    "?",
    "!"
]

class Vocab(object):
    def __init__(self, dataset='Clevr', with_SE=True):
        vocab = vocab_clevr_list

        #with open(filename) as f:
        if with_SE:
            self.itos = [PAD, BOS, EOS, UNK] + vocab #[ token.strip() for token in f.readlines() ]
        else:
            self.itos = [PAD, UNK] + vocab #[ token.strip() for token in f.readlines() ]
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self._size = len(self.stoi)
        self._padding_idx = self.stoi[PAD]
        self._unk_idx = self.stoi[UNK]
        self._start_idx = self.stoi.get(BOS, -1)
        self._end_idx = self.stoi.get(EOS, -1)

    def random_sample(self):
        return self.idx2token(1 + np.random.randint(self._size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self.itos[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self.stoi.get(x, self.unk_idx)

    @property
    def size(self):
        return self._size

    @property
    def padding_idx(self):
        return self._padding_idx

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx

def ListsToTensor(xs, vocab, with_S=True, with_E=True, mx_len=50):
    batch_size = len(xs)
    for i in range(batch_size):
        cur_len = len(xs[i])
        xs[i] = xs[i][:min(cur_len, mx_len)]

    lens = [len(x) + (1 if with_S else 0) + (1 if with_E else 0) for x in xs]
    #mx_len = max(max(lens),1)
    ys = []
    for i, x in enumerate(xs):
        y = ([vocab.start_idx] if with_S else [] )+ [vocab.token2idx(w) for w in x] + ([vocab.end_idx] if with_E else []) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    lens = np.array([ max(1, x) for x in lens])
    data = np.array(ys) #np.transpose(np.array(ys))
    return data, lens

def getTextLists(x, with_S=True, with_E=True, mx_len=50):
    x = x[:min(mx_len, len(x))]
    x_len = len(x) + (1 if with_S else 0) + (1 if with_E else 0)
    x = ([BOS] if with_S else [] )+ x + ([EOS] if with_E else []) + ([PAD]*(mx_len - x_len))
    return x, x_len

if __name__ == "__main__":
    pass

