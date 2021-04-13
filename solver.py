# Author: Yahui Liu <yahui.liu@unitn.it>
# Gaussian Mixture Models for Unsupervised Image-to-Image Translation 
#   Conditioned on Text

import os
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable

from vocab import Vocab
from tools import dist_sampling_split
from networks.networks import MsImageDis
from networks.networks_v2 import AdaINGen_v2
from gmm import gmm_kl_distance, gmm_kl_distance_sp, gmm_earth_mover_distance_sp
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler, moving_average

class Solver(nn.Module):
    def __init__(self, configs, device=None, pretrained_embed=None):
        super(Solver, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.configs =  configs

        self.vocab = Vocab(dataset=configs['dataset'])
        #print('self.vocab', self.vocab.itos)
        #stop
        
        # Initiate the networks
        self.gen = AdaINGen_v2(configs['input_dim'], self.vocab, configs['gen'], 
            pretrained_embed=pretrained_embed)  # auto-encoder for domain a
        self.dis = MsImageDis(configs['input_dim'], configs['dis'], self.device)  # discriminator for domain a
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.print_network(self.dis, 'D')
        self.print_network(self.gen, 'G')

        self.num_cls = configs['gen']['num_cls']
        self.c_dim = configs['c_dim']
        #self.style_dim = self.num_cls*self.c_dim

        self.dist_mode = configs['dist_mode']
        self.use_attention = configs['gen']['use_attention']
        self.att_status = self.use_attention
        self.ds_iter = configs['ds_iter']

        # fix the noise used in sampling
        display_size = int(configs['display_size'])
        self.display_size = display_size

        self.dataset = configs['dataset']
        self.stddev  = configs['stddev']
        self.sigma   = torch.tensor(self.stddev**2).to(self.device)
        self.d_reg_every = 16
        self.rnd_step = 3
        self.init_ds_w = configs['ds_w']

        lr = configs['lr']
        self.lr_policy = configs['lr_policy']

        # Setup the optimizers
        beta1, beta2 = configs['beta1'], configs['beta2']
        
        total_params = sum(p.numel() for p in self.gen.parameters())
        print("total params before", total_params)
        
        trainable_params = sum(p.numel() for p in self.gen.parameters() if p.requires_grad)
        print("trainable params with bert", trainable_params)
        
        #set BERT off
        for param in self.gen.enc_txt.bert.parameters():
            param.requires_grad = False
            
        #print("self.gen.enc_txt.bert.parameters()", self.gen.enc_txt.bert.parameters)
        
        #STOP
            
        trainable_params = sum(p.numel() for p in self.gen.parameters() if p.requires_grad)
        print("trainable params exclude bert", trainable_params)
        
        #stop
        
        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters())
        
        
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=configs['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=configs['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, configs)
        self.gen_scheduler = get_scheduler(self.gen_opt, configs)

        # Network weight initialization
        self.apply(weights_init(configs['init']))
        self.dis.apply(weights_init('gaussian'))

        self.criterionL1 = torch.nn.L1Loss()

        # Load VGG model if needed
        if 'vgg_w' in configs.keys() and configs['vgg_w'] > 0:
            #self.vgg = load_vgg16(configs['vgg_model_path'] + '/models').to(device)
            self.vgg = load_vgg16('models').to(device)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of parameters in {}: {}".format(name, num_params))

    def copy_nets(self):
        self.gen_copy = copy.deepcopy(self.gen)
        self.dis_copy = copy.deepcopy(self.dis)

    def update_learning_rate(self):
        if self.lr_policy == 'cosa':
            if self.dis_opt.param_groups[0]['lr'] == self.configs['eta_min'] or \
                self.gen_opt.param_groups[0]['lr'] == self.configs['eta_min']:
                self.configs['step_size'] *= self.configs['t_mult']
                self.dis_scheduler = get_scheduler(self.dis_opt, self.configs)
                self.gen_scheduler = get_scheduler(self.gen_opt, self.configs)

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def update_attention_status(self, iters):
        if self.att_status:
            self.use_attention = False if iters < 10000 else True

    def recon_criterion(self, x, y):
        return torch.mean(torch.abs(x - y))

    def distance(self, x, y):
        #return torch.sqrt(torch.sum((x-y)**2, dim=1))
        return torch.mean(torch.abs(x-y).sum(dim=1))

    def isometry_constraint(self, z1, z2, rec_z1, rec_z2):
        return torch.abs(self.distance(z1, z2) - self.distance(rec_z1, rec_z2)).mean()

    def mode_seeking_constraint(self, im1, im2, z1, z2, eps=1e-5):
        loss = torch.mean(torch.abs(im1 - im2)) / torch.mean(torch.abs(z1 - z2))
        return 1.0 / (loss + eps)

    def criterion_l1(self, a, z):
        if isinstance(a, list):
            a = torch.cat(a, dim=1)
        if isinstance(z, list):
            z = torch.cat(z, dim=1)
        return self.criterionL1(a, z)

    def style_replace(self, c_src, c_trg, z_src, z_trg):
        mark = c_src==c_trg
        for i in range(c_src.size(0)):
            for j in range(c_src.size(1)):
                if mark[i,j]:
                    z_trg[i, j*self.c_dim:(j+1)*self.c_dim] = z_src[i, j*self.c_dim:(j+1)*self.c_dim].clone()
        return z_trg

    def forward(self, x_real, txt_src2trg, testing_bert, config_wb, txt_lens):
        content, style_src, _ = self.gen.encode(x_real)
        style_txt, _ = self.gen.encode_txt(torch.cat(style_src, dim=1), txt_src2trg,testing_bert, config_wb, txt_lens)

        x_fake, x_fake_att = self.gen.decode(content, style_txt)
        if self.use_attention:
            x_fake = x_fake * x_fake_att  + x_real * (1-x_fake_att) 
        return x_fake

    def gen_update(self, x_real, c_src, c_trg, txt_src2trg, testing_bert, config_wb, txt_lens, 
        label_src, label_trg, configs, iters):
        print("txt_lens for gen_update in solver", txt_lens)
        
        self.gen_opt.zero_grad()
        # encode
        content_real, style_real, logvar = self.gen.encode(x_real)
        
        # decode (within domain)
        x_real_rec, x_real_rec_att = self.gen.decode(content_real, 
            torch.cat(style_real,dim=1))
        if self.use_attention:
            x_real_rec = x_real_rec*x_real_rec_att + x_real*(1-x_real_rec_att)
            
            ''' This is where to implement 2-sided attention for within-domain'''
            
            
        content_real_rec, style_real_rec, _ = self.gen.encode(x_real_rec)
        
        # decode (cross domain)
        style_txt, logvar_txt = self.gen.encode_txt(torch.cat(style_real, dim=1), 
            txt_src2trg, testing_bert, config_wb, txt_lens)
        
        x_fake, x_fake_att = self.gen.decode(content_real, 
            torch.cat(style_txt,dim=1))
        
        
        if self.use_attention:
            x_fake = x_fake*x_fake_att + x_real*(1-x_fake_att)
            
            ''' This is where to implement 2-sided attention for cross-domain'''
        
        #self.loss_ds = 0.0
        #if self.stddev > 0 and iters > self.ds_iter:
        style1 = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)
        x_fake1, x_fake_att1 = self.gen.decode(content_real, style1)
        
        style2 = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)
        x_fake2, x_fake_att2 = self.gen.decode(content_real, style2)
        
        if self.use_attention:
            x_fake1 = x_fake1*x_fake_att1 + x_real*(1-x_fake_att1)
            x_fake2 = x_fake2*x_fake_att2 + x_real*(1-x_fake_att2)
            
        self.loss_ds = torch.mean(torch.abs(x_fake1 - x_fake2.detach()))
        content_rand, style_rand, _ = self.gen.encode(x_fake1)
        self.init_ds_w = max(self.init_ds_w-1/1e5, 0.0)
        
        # encode again
        content_fake_rec, style_fake_rec, _ = self.gen.encode(x_fake)
        # decode again (if needed)
        if configs['recon_x_cyc_w'] > 0:
            x_cycle, x_cycle_att = self.gen.decode(content_fake_rec, 
                torch.cat(style_real,dim=1))
            if self.use_attention:
                x_cycle = x_cycle*x_cycle_att + x_real*(1-x_cycle_att)

        # reconstruction loss
        self.loss_gen_recon_x  = self.recon_criterion(x_real_rec, x_real)
        self.loss_gen_recon_c_real = self.recon_criterion(content_real_rec, content_real)
        self.loss_gen_recon_c_fake = self.recon_criterion(content_fake_rec, content_real)
        self.loss_gen_recon_c_rand = self.recon_criterion(content_rand, content_real)
        self.loss_gen_recon_s_real = self.criterion_l1(style_real_rec, style_real)
        self.loss_gen_recon_s_fake = self.criterion_l1(style_fake_rec, style_txt)
        self.loss_gen_recon_s_rand = self.criterion_l1(style_rand, style1)
        
        self.loss_gen_cycrecon_x = 0
        if configs['recon_x_cyc_w'] > 0:
            self.loss_gen_cycrecon_x = self.recon_criterion(x_cycle, x_real)

        # GAN loss
        self.loss_gen_adv = self.dis.calc_gen_loss(x_fake, label_trg, configs['gan_w'], configs['cls_w']) + \
            self.dis.calc_gen_loss(x_fake1, label_trg, configs['gan_w'], configs['cls_w'])

        # KL loss
        self.loss_kl_x, self.loss_kl_trg = 0.0, 0.0
        if self.dist_mode == 'kls':
            self.loss_kl_x = gmm_kl_distance_sp(style_real, logvar, c_src, self.sigma)
            self.loss_kl_trg = gmm_kl_distance_sp(style_txt, logvar_txt, c_trg, self.sigma)
        else: #  self.dist_mode == 'em':
            self.loss_kl_x = gmm_earth_mover_distance_sp(style_real, c_src)
            self.loss_kl_trg = gmm_earth_mover_distance_sp(style_txt, c_trg)

        # domain-invariant perceptual loss
        self.loss_gen_vgg = 0
        if configs['recon_x_cyc_w'] > 0 and configs['vgg_w'] > 0:
            self.loss_gen_vgg = self.compute_vgg_loss(self.vgg, x_real, x_cycle)

        # total loss
        self.loss_gen_total = self.loss_gen_adv + \
                              configs['recon_x_w'] * self.loss_gen_recon_x + \
                              configs['recon_c_w'] * self.loss_gen_recon_c_real + \
                              configs['recon_c_w'] * self.loss_gen_recon_c_fake + \
                              configs['recon_c_w'] * self.loss_gen_recon_c_rand + \
                              configs['recon_s_w'] * self.loss_gen_recon_s_real + \
                              configs['recon_s_w'] * self.loss_gen_recon_s_fake + \
                              configs['recon_s_w'] * self.loss_gen_recon_s_rand + \
                              configs['recon_x_cyc_w'] * self.loss_gen_cycrecon_x + \
                              configs['kl_w'] * self.loss_kl_x + \
                              configs['kl_w'] * self.loss_kl_trg + \
                              configs['vgg_w'] * self.loss_gen_vgg - \
                              self.init_ds_w * self.loss_ds
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.device)
        target_vgg = vgg_preprocess(target, self.device)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_real, txt_src2trg, testing_bert, config_wb, txt_lens):
        self.eval()
        x_real_recon, x_ab, x_sam, x_att = [], [], [], []
        for i in range(x_real.size(0)):
            content_real, style_real, _ = self.gen.encode(x_real[i:i+1])
            style_real = torch.cat(style_real, dim=1)
            style_txt, logvar_txt = self.gen.encode_txt(style_real, 
                txt_src2trg[i:i+1], testing_bert, config_wb, txt_lens[i:i+1])
            style_txt = torch.cat(style_txt,dim=1)
            
            x_real_rec, x_real_rec_att = self.gen.decode(content_real, style_real)
            x_trg, x_trg_att = self.gen.decode(content_real, style_txt)

            mus_real = torch.ones(1,self.num_cls).float().to(self.device)
            mus_txt = torch.ones(1,self.num_cls).float().to(self.device)
            for idx in range(self.num_cls):
                if style_real[0,idx*self.c_dim:(idx+1)*self.c_dim].mean() < 0.0:
                    mus_real[0,idx] = -1.0
                if style_txt[0,idx*self.c_dim:(idx+1)*self.c_dim].mean() < 0.0:
                    mus_txt[0,idx] = -1.0
            z_sample = dist_sampling_split(mus_txt, self.c_dim, self.stddev, self.device)
            z_sample = self.style_replace(mus_real, mus_txt, style_real, z_sample)
            x_sample, x_sample_att = self.gen.decode(content_real, z_sample)

            if self.use_attention:
                x_trg = x_trg*x_trg_att + x_real[i:i+1]*(1-x_trg_att)
                x_real_rec = x_real_rec*x_real_rec_att + x_real[i:i+1]*(1-x_real_rec_att)
                x_sample = x_sample*x_sample_att + x_real[i:i+1]*(1-x_sample_att)
                x_att.append(torch.cat([x_trg_att, x_trg_att, x_trg_att],dim=1))
            x_ab.append(x_trg)
            x_real_recon.append(x_real_rec)
            x_sam.append(x_sample)
        x_real_recon = torch.cat(x_real_recon)
        x_ab = torch.cat(x_ab)
        x_sam = torch.cat(x_sam)
        
        # x_real - source image
        # x_real_recon - reconstructed source image (for content & visual attr constrain), 
        # x_ab - manipulated image
        # x_sam - sampled domain (for diversity)
   
        outputs = [x_real, x_real_recon, x_ab, x_sam]
        
        if self.use_attention:
            x_att = torch.cat(x_att)
            outputs.append((x_att-0.5)/0.5)
        self.train()
        return outputs

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def r1_penalty(self, y, x):
        # gradient penalty
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(y, x,
                                   grad_outputs=weight,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2sqr = torch.sum(dydx**2, dim=1)
        r2_penalty = torch.mean(dydx_l2sqr**2)
        return r2_penalty

    def dis_update(self, x_real, c_src, c_trg, txt_src2trg, testing_bert, config_wb, txt_lens, 
        label_src, label_trg, configs, iters):
        self.dis_opt.zero_grad()
        content_real, style_real, _ = self.gen.encode(x_real)
        style_real = torch.cat(style_real, dim=1)

        style1 = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)
        style_txt, logvar_txt = self.gen.encode_txt(style_real, 
            txt_src2trg, testing_bert, config_wb, txt_lens)
        style_txt = torch.cat(style_txt, dim=1)
        x_fake, x_fake_att = self.gen.decode(content_real, style_txt)
        x_fake1, x_fake_att1 = self.gen.decode(content_real, style1)
        if self.use_attention:
            x_fake = x_fake*x_fake_att + x_real*(1-x_fake_att)
            x_fake1 = x_fake1*x_fake_att1 + x_real*(1-x_fake_att1)

        self.loss_dis = self.dis.calc_dis_loss(x_fake, x_real, label_trg, label_src, configs['gan_w'], configs['cls_w']) + \
            self.dis.calc_dis_loss(x_fake1, x_real, label_trg, label_src, configs['gan_w'], configs['cls_w'])
        self.loss_dis_all = self.loss_dis

        # Compute loss for gradient penalty.
        if configs['gp_w'] > 0.0:
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.dis(x_hat, False)[0]
            self.loss_gp = self.gradient_penalty(out_src, x_hat) * configs['gp_w']
            self.loss_dis_all += self.loss_gp

        # Compute loss for r1 penalty.
        if configs['use_r1'] and (iters+1) % self.d_reg_every == 0:
            x_real.requires_grad = True
            output, _  = self.dis(x_real, False)[0]
            self.loss_r1 = self.r1_penalty(output, x_real) * 10. / 2  #* self.d_reg_every 
            self.loss_dis_all += self.loss_r1

        self.loss_dis_all.backward()
        self.dis_opt.step()

    def smooth_moving(self):
        moving_average(self.gen, self.gen_copy)
        moving_average(self.dis, self.dis_copy)

    def resume(self, checkpoint_dir, configs):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        print("last_model_name", last_model_name[2:])
        
        state_dict = torch.load(last_model_name[2:], map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-15:-7]) if 'avg' in last_model_name else int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name, map_location=lambda storage, loc: storage)
        self.dis.load_state_dict(state_dict['b'])
        # Load optimizers
        #state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'), map_location=lambda storage, loc: storage)
        #self.dis_opt.load_state_dict(state_dict['dis'])
        #self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, configs, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, configs, iterations)
        if torch.__version__ != '0.4.1': 
            for _ in range(iterations):
                self.gen_scheduler.step()
                self.dis_scheduler.step()
        print('Resume from iteration %d' % iterations)
        return iterations

    def init_network(self, gen_path, dis_path):
        """In order to tuning the models with CAMs"""
        gen_dict = torch.load(gen_path, map_location=lambda storage, loc: storage)['a']
        dis_dict = torch.load(dis_path, map_location=lambda storage, loc: storage)['b']
        gen_state_dict = self.gen.state_dict()
        dis_state_dict = self.dis.state_dict()

        for key in dis_state_dict:
            if key in dis_dict:
                dis_state_dict[key] = dis_dict[key]
        self.dis.load_state_dict(dis_state_dict)

        for key in gen_state_dict:
            if key in gen_dict and 'embed_tokens' not in key:
                gen_state_dict[key] = gen_dict[key]
        self.gen.load_state_dict(gen_state_dict)

        print("Initial model loaded...")

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        gen_copy_name = os.path.join(snapshot_dir, 'gen_%08d_avg.pt' % (iterations + 1))
        dis_copy_name = os.path.join(snapshot_dir, 'dis_%08d_avg.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen.state_dict()}, gen_name)
        torch.save({'b': self.dis.state_dict()}, dis_name)
        torch.save({'a': self.gen_copy.state_dict()}, gen_copy_name)
        torch.save({'b': self.dis_copy.state_dict()}, dis_copy_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
