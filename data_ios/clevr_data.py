import json
import numpy as np
from PIL import Image
from os.path import join

import torch
from torch.utils import data
from torchvision import transforms as T

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vocab import Vocab, ListsToTensor


class CLEVR(data.Dataset):
    """Dataset class for the CLEVR dataset."""
    def __init__(self, dataroot:str, annotations:str, mode:str, transform):
        self.img_dir = join(dataroot, mode)
        self.data = json.load(open(annotations, 'r'))
        self.transform = transform
        self.vocab = Vocab(dataset='Clevr')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]

        img_fname = join(self.img_dir, data_point['im'])
        image = Image.open(img_fname).convert('RGB')
        image = self.transform(image)

        src_attr = torch.tensor(data_point['a']).float()
        trg_attr = torch.tensor(data_point['ta']).float()
        if trg_attr.shape[0] > 1:
            trg_idx = np.random.randint(trg_attr.shape[0])
            trg_attr = trg_attr[trg_idx]
        trg_attr = trg_attr.squeeze()

        cmd = data_point['cd']
        cmd_tensor, txt_lens = ListsToTensor([cmd.split()], self.vocab, mx_len=80)
        cmd_tensor = torch.from_numpy(cmd_tensor).squeeze(0).long()
        txt_lens = torch.from_numpy(txt_lens).squeeze(0).long()

        return image, src_attr, trg_attr, cmd_tensor, txt_lens, cmd


if __name__=="__main__":
    transform = []
    transform.append(T.Resize((128, 128)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = CLEVR(dataroot='/media/namrata/Data/data/CLEVR_v1.0/images', 
                    annotations='/media/namrata/Data/data/CLEVR_val_data_full.json', 
                    mode='val', transform=transform)
    dataset[0]