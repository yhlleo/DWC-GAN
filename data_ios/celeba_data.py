"""
Base on: https://github.com/yunjey/stargan
"""

import os
import sys
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data

from .celeba_text import labels2text as lab2txt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vocab import Vocab, ListsToTensor

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset  = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.vocab = Vocab(dataset='CelebA')
        self.all_domains = self.collect_all_domains(len(selected_attrs))

        self.labels2text = lab2txt

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
        print(selected_attrs)
        print(self.num_images)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        target_strings = []
        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(int(values[idx] == '1'))

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def collect_all_domains(self, num_attr):
        domains = ['0', '1']
        for i in range(1, num_attr):
            current_domains = []
            for da in domains:
                current_domains.extend([da+'0', da+'1'])
            domains = current_domains

        domains_str2int = []
        for da in domains:
            domains_str2int.append([int(v) for v in da])
        return domains_str2int

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, src_label = dataset[index]
        _, trg_label = random.choice(dataset)
        #trg_label = random.choice(self.all_domains)

        # -------------- labels to text -------------- #
        #print(src_label, trg_label)
        diff_txt = self.labels2text(np.array(src_label), np.array(trg_label))
        txt2tensor, txt_lens = ListsToTensor([diff_txt.split()], self.vocab, mx_len=80)
        txt2tensor = torch.from_numpy(txt2tensor).squeeze(0).long()
        txt_lens   = torch.from_numpy(txt_lens).squeeze(0).long()
        # -------------------------------------------- #
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        image = self.transform(image)
        if image.size(0) == 1: # convert grayscale to rgb
            image = torch.cat([image, image, image], dim=0)

        src_label = torch.tensor(src_label).float()
        trg_label = torch.tensor(trg_label).float()
        return image, src_label, trg_label, txt2tensor, txt_lens

    def __len__(self):
        """Return the number of images."""
        return self.num_images

if __name__ == '__main__':
    celeba_dataset = CelebA('./datasets/celeba/images',
                            './datasets/celeba/list_attr_celeba-v2.txt',
                            ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Smiling', 'Young', 'Male', 'Eyeglasses', 'No_Beard'],
                            None, 'train')
    length = len(celeba_dataset)
    dataset = celeba_dataset.train_dataset
    with open('celeba_demo.txt', 'w') as fin:
        for num in range(3): # three epochs
            for i in range(length):
                _, src_label = dataset[i]
                _, trg_label = random.choice(dataset)
                diff_txt = lab2txt(np.array(src_label), np.array(trg_label))
                src_label = [str(v) for v in src_label]
                trg_label = [str(v) for v in trg_label]
                fin.write(''.join(src_label)+ ' ' + ''.join(trg_label) + ' ' + diff_txt+'\n')

