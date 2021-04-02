import json
import numpy as np
from PIL import Image
from os.path import join

import torch
from torch.utils import data
from torchvision import transforms

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clevr_vocab import Vocab, ListsToTensor


class CLEVR(data.Dataset):
	"""Dataset class for the CLEVR dataset."""
	def __init__(self, dataroot:str, annotations:str, mode:str):
		self.img_dir = join(dataroot, mode)
		self.data = json.load(open(annotations, 'r'))
		self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		self.vocab = Vocab(dataset='Clevr')

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		data_point = self.data[index]

		img_fname = join(self.img_dir, data_point['im'])
		image = Image.open(img_fname).convert('RGB')
		image = torch.FloatTensor(np.array(image)) / 255.
		image = self.normalize(image.permute(2, 0, 1))

		src_attr = torch.tensor(data_point['a']).float()
		trg_attr = torch.tensor(data_point['ta']).float()

		cmd = data_point['cd']
		cmd_tensor, txt_lens = ListsToTensor([cmd.split()], self.vocab, mx_len=80)
		cmd_tensor = torch.from_numpy(cmd_tensor).squeeze(0).long()
		txt_lens = torch.from_numpy(txt_lens).squeeze(0).long()

		return image, src_attr, trg_attr, cmd_tensor, txt_lens


if __name__=="__main__":
	dataset = CLEVR(dataroot='/media/namrata/Data/data/CLEVR_v1.0/images', 
					annotations='../datasets/clevr/CLEVR_train_sample_small.json', 
					mode='train')
	dataset[0]