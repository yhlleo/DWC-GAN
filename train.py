"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
import shutil
import pickle
import argparse

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import tensorboardX

## Dan
from data_ios.celeba_text import labels2text as lab2txt
from vocab import Vocab, ListsToTensor
##


from tools import asign_label
from data_loader import get_loader
from solver import Solver
from utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images_single, Timer


import math
import random
import wandb


#print('***********************************************')

#stop now0

cudnn.benchmark = True
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", type=int, default=0)
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu list')
parser.add_argument('--use_pretrained_embed', type=int, default=0)
parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
opts = parser.parse_args()

#print( "gpu_id",opts.gpu_ids)
#STOP



# 1️⃣ Start a new run, tracking config metadata
wandb.init(project="training-BERT", config={
    "approach": "BERT last hidden",
    "dropout": 0.1,
    "use_bert": True,
    "use_VAE":False,
    "use_pretrained_embed" : opts.use_pretrained_embed,
    "use_pretrain_model": opts.resume,
    "n_critic": opts.n_critic,
    "gpu_id" :opts.gpu_ids,
    "resume": opts.resume,
    "output_path": opts.output_path,
    
    "gen_kl_weight2":  0.1,
    "gen_kl_weight3":  0.3,
    "gen_kl_weight4":  0.3,
    "gen_kl_weight5":  0.3,
    
})
config_wb = wandb.config





# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
dataset_name = config['dataset']
# get device name: CPU or GPU
print('opts.gpu_ids',opts.gpu_ids)
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')
print('device', device)

print('torch available', torch.cuda.is_available())

if opts.n_critic < 1: 
    opts.n_critic = 1 
attr_path = config['attr_path'] if 'attr_path' in config else None

selected_attrs = None
if dataset_name == "CelebA":
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male',  
                      'Smiling', 'Young',  'Eyeglasses', 'No_Beard']

train_loader = get_loader(config['data_root'], config['crop_size'], config['image_size'], 
    config['batch_size'], attr_path, selected_attrs, dataset_name, 'train', config['num_workers'])

test_loader  = get_loader(config['data_root'], config['crop_size'], config['image_size'], 
    1, attr_path, selected_attrs, dataset_name, 'test', config['num_workers'])

#stop now1

#print('display_size in train.py', display_size)

train_display        = [train_loader.dataset[i] for i in range(display_size)]
train_display_images = torch.stack([item[0] for item in train_display]).to(device)
test_display         = [test_loader.dataset[i] for i in range(display_size)]
test_display_images  = torch.stack([item[0] for item in test_display]).to(device)

train_display_txt    = torch.stack([item[3] for item in train_display]).to(device)
train_display_txt_lens = torch.stack([item[4] for item in train_display]).to(device)
test_display_txt    = torch.stack([item[3] for item in test_display]).to(device)
test_display_txt_lens = torch.stack([item[4] for item in test_display]).to(device)

pretrained_embed=None
# if opts.use_pretrained_embed:
if config_wb.use_pretrained_embed:
    with open(config['pretrained_embed'], 'rb') as fin:
        pretrained_embed = pickle.load(fin)
        
        
# Setup model and data loader
trainer = Solver(config, device, pretrained_embed).to(device)
# if config['use_pretrain']:
if config_wb.use_pretrain_model:
    trainer.init_network(config['gen_pretrain'], config['dis_pretrain'])

    
    
# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, config) if opts.resume else 0
trainer.copy_nets()





def ind2text(text,  dict_rev):
    result = []
    for i in text:


        if dict_rev[i] == '<unk>':
            #print(dict_rev[i])
            result.append('[UNK]')

        elif dict_rev[i] in ["unsmile", "unsmiling",]:
            print(dict_rev[i])
            result.append('un')
            result.append(dict_rev[i][2:])

        elif i not in [0,1,2]:
            #print(dict_rev[i])
            result.append(dict_rev[i])

    return ' '.join(result)




while True:
    for it, data_iter in enumerate(train_loader):
        print("$$$$$$$$$$$$$$$$$$$$$$$ ITERATION {} $$$$$$$$$$$$$$$$$$$".format(it))
        wandb.log({"Iteration": it})
        
        #if config['dataset'] == 'CelebA':
        x_real, label_src, label_trg, txt_src2trg, txt_lens = data_iter
        
        print('START')
        print('----------------------------txt_lens in data loader in train.py', txt_lens, '----------------')
        
        #print('----------------------------txt_src2trg in data loader in train.py', txt_src2trg, '----------------')
        #print(' IN BETWEEN' )
        #print('----------------------------label_src in data loader in train.py', label_src)
        #print('----------------------------label_trg in data loader in train.py', label_trg)
        
        vocab = Vocab(dataset='CelebA')
        


        from collections import defaultdict
        vocabulary = vocab.stoi
        
        vocab_rev = defaultdict()
        for k,v in vocabulary.items():
            vocab_rev[v] = k
            
        test = txt_src2trg.view(-1).tolist()

        testing_bert = ind2text(test, vocab_rev)
        print('testing_bert:', testing_bert)
        
        
        #print('*********REVERSED TEXT in train.py ---------> {}'.format(rev_text))
        
        
        print('STOP')
        

        
        c_src = asign_label(label_src, config['c_dim'], dataset_name).to(device)# e.g [0,0,1,1] -> [-1,-1,1,1]
        c_trg = asign_label(label_trg, config['c_dim'], dataset_name).to(device)
        
#         print("c_src", c_src)
#         print("c_trg", c_trg)
#         stop now

        x_real = x_real.to(device)
        label_src = label_src.to(device)
        label_trg = label_trg.to(device)
        txt_src2trg = txt_src2trg.to(device)
        txt_lens = txt_lens.to(device)
        
        with Timer("Elapsed time in update: %f"):
            trainer.dis_update(x_real, c_src, c_trg, txt_src2trg, testing_bert, config_wb, txt_lens, label_src, 
                label_trg, config, iterations)
            if (iterations+1) % opts.n_critic == 0:
                trainer.gen_update(x_real, c_src, c_trg, txt_src2trg, testing_bert, config_wb, txt_lens, 
                    label_src, label_trg, config, iterations)
            torch.cuda.synchronize()
            trainer.smooth_moving()
        trainer.update_learning_rate()
        trainer.update_attention_status(iterations)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            print('Loss: gen %.04f, dis %.04f' % (trainer.loss_gen_total.data, trainer.loss_dis_all.data))
            write_loss(iterations, trainer, train_writer)
            print('Iter {}, lr {}, ds {}'.format(iterations, trainer.gen_opt.param_groups[0]['lr'], trainer.init_ds_w))
            
            
            
            # 2️⃣ Log metrics from your script to W&B
            wandb.log({"gen loss":trainer.loss_gen_total.data, "dis loss":trainer.loss_dis_all.data})


        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images, 
                    test_display_txt, testing_bert, config_wb, test_display_txt_lens)
                train_image_outputs = trainer.sample(train_display_images, 
                    train_display_txt, testing_bert,config_wb, train_display_txt_lens)
            write_2images_single(test_image_outputs, display_size, 
                image_directory, 'test_%08d' % (iterations + 1))
            write_2images_single(train_image_outputs, display_size, 
                image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, 
                config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images, 
                    train_display_txt, testing_bert, config_wb, train_display_txt_lens)
            write_2images_single(image_outputs, display_size, 
                image_directory, 'train_current')
            
            
        
        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            
            wandb.finish()
            sys.exit('Finish training')

