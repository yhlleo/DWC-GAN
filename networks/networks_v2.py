
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .networks import ContentEncoder, MLP, Conv2dBlock, ResBlocks, AdaptiveInstanceNorm2d


class AdaINGen_v2(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, vocab, params, pretrained_embed=None):
        super(AdaINGen_v2, self).__init__()
        dim           = params['dim']
        n_res         = params['n_res']
        activ         = params['activ']
        pad_type      = params['pad_type']
        mlp_dim       = params['mlp_dim']
        use_attention = params['use_attention']
        c_dim         = params['c_dim']
        num_cls       = params['num_cls']

        embed_dim     = params['embed_dim']
        hidden_size   = params['hidden_size']
        num_layers    = params['num_layers']
        dropout_in    = params['dropout_in']
        dropout_out   = params['dropout_out']

        style_dim     = c_dim * num_cls
        style_downsample   = params['style_downsample']
        content_downsample = params['content_downsample']
        use_map       = params['use_map']
        #dataset       = params['dataset']

        # style encoder
        self.enc_style = StyleEncoder(style_downsample, input_dim, dim, 
            norm='none', activ=activ, pad_type=pad_type, c_dim=c_dim, 
            num_class=num_cls, use_map=use_map)

        # content encoder
        self.enc_content = ContentEncoder(content_downsample, n_res, 
            input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(content_downsample, n_res, self.enc_content.output_dim, 
            input_dim, res_norm='adain', activ=activ, pad_type=pad_type, 
            use_attention=use_attention)

        # style transfer
        self.enc_txt = TxtEncoder(vocab, embed_dim, hidden_size, c_dim, 
            num_cls, num_layers, dropout_in, dropout_out, 
            pretrained_embed=pretrained_embed)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, 
            norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, mus, logvar = self.encode(images)
        images_recon = self.decode(content, mus)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        mus, logvar = self.enc_style(images)
        content = self.enc_content(images)
        return content, mus, logvar

    def encode_txt(self, style_ord, txt_org2trg, txt_lens):
        mu, logvar = self.enc_txt(style_ord, txt_org2trg, txt_lens)
        return mu, logvar

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, norm, 
        activ, pad_type, c_dim, num_class, use_map=False):
        super(StyleEncoder, self).__init__()
        self.num_class = num_class
        self.use_map   = use_map
        
        model = []
        model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        
        for i in range(2):
            model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *=2
        for i in range(n_downsample - 2):
            model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model = nn.Sequential(*model)

        if self.use_map:
            self.mapping = nn.Sequential(nn.Linear(dim, dim),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.1),
                                         nn.Linear(dim, dim),
                                         nn.ReLU(inplace=True))

        self.fcs = nn.ModuleList()
        self.fcvars = nn.ModuleList()
        for i in range(self.num_class):
            self.fcs.append(nn.Linear(dim, c_dim))
            self.fcvars.append(nn.Linear(dim, c_dim))
        self.output_dim = dim

    def forward(self, x):
        feats  = self.model(x)
        feats_ = feats.view(x.size(0), -1)

        if self.use_map:
            feats_ = self.mapping(feats_)

        fcs, fcvars = [], []
        for i in range(self.num_class):
            fcs.append(self.fcs[i](feats_))
            fcvars.append(self.fcvars[i](feats_))
        return fcs, fcvars


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', 
        activ='relu', pad_type='zero', use_attention=False):
        super(Decoder, self).__init__()
        self.use_attention = use_attention
        model = []
        # AdaIN residual blocks
        model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model = nn.Sequential(*model)
        self.image_content   = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.image_attention = Conv2dBlock(dim, 1, 7, 1, 3, norm='none', activation='sigmoid', pad_type=pad_type)

    def forward(self, x):
        feats     = self.model(x)
        content   = self.image_content(feats)
        attention = None

        if self.use_attention:
            attention = self.image_attention(feats)
        return content, attention

class TxtEncoder(nn.Module):
    def __init__(self, vocab, embed_dim=512, hidden_size=512, 
        c_dim=8, num_class=8, num_layers=1, dropout_in=0.1, 
        dropout_out=0.1, bidirectional=True, pretrained_embed=None):
        super(TxtEncoder, self).__init__()
        self.vocab       = vocab
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_in  = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.num_class   = num_class
        self.style_dim   = c_dim * num_class

        self.embed_tokens = nn.Embedding(vocab.size, embed_dim, vocab.padding_idx)
        if pretrained_embed is not None:
            weights_matrix = np.zeros((vocab.size, embed_dim))
            for i, word in enumerate(vocab.itos):
                try: 
                    weights_matrix[i] = pretrained_embed[word]
                except KeyError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(embed_dim,))
            self.embed_tokens.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
            self.embed_tokens.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_dim+self.style_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )
        hidden_dim = hidden_size*num_layers
        hidden_dim *= 4 if bidirectional else 2
        self.fcs = nn.ModuleList()
        self.fcvars = nn.ModuleList()
        for i in range(self.num_class):
            self.fcs.append(nn.Linear(hidden_dim, c_dim))
            self.fcvars.append(nn.Linear(hidden_dim, c_dim))

    # Using raw text as input
    def forward(self, style_ord, src_tokens, src_lengths):
        src_tokens = src_tokens.transpose(1,0)
        seq_len, bsz = src_tokens.size()
        ###
        sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)
        sorted_src_tokens = src_tokens.index_select(1, indices)
        sorted_style_ord = style_ord.index_select(0, indices)
        ###
        x = self.embed_tokens(sorted_src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = torch.cat([x, sorted_style_ord.expand(seq_len, -1, -1)], -1)
        
        packed_x = nn.utils.rnn.pack_padded_sequence(x, sorted_src_lengths.data.tolist())

        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_h, final_c) = self.lstm(packed_x, (h0, c0))

        mem, _ = nn.utils.rnn.pad_packed_sequence(packed_outs)
        mem = F.dropout(mem, p=self.dropout_out, training=self.training)

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)
            final_h = combine_bidir(final_h)
            final_c = combine_bidir(final_c)

        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions) # num_layers x bsz x hidden_size
        final_c = final_c.index_select(1, positions) 

        batch_size = final_h.size(1)
        output = torch.cat([final_h, final_c], dim=1).view(batch_size, -1)
        fcs, fcvars = [], []
        for i in range(self.num_class):
            fcs.append(self.fcs[i](output))
            fcvars.append(self.fcvars[i](output))
        return fcs, fcvars

    # Using word embeddings as input directly
    def forward_embed(self, style_ord, embeddings, lengths):
        _, indices = torch.sort(lengths, descending=True)
        x = embeddings.transpose(1,0) # Batch_size,Seq_len,Dim -> S,B,D
        seq_len, bsz, emb_dim = x.size()
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = torch.cat([x, style_ord.expand(seq_len, -1, -1)], -1)
        
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)

        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_h, final_c) = self.lstm(packed_x, (h0, c0))

        mem, _ = nn.utils.rnn.pad_packed_sequence(packed_outs)
        mem = F.dropout(mem, p=self.dropout_out, training=self.training)

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)
            final_h = combine_bidir(final_h)
            final_c = combine_bidir(final_c)

        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions) # num_layers x bsz x hidden_size
        final_c = final_c.index_select(1, positions) 

        batch_size = final_h.size(1)
        output = torch.cat([final_h, final_c], dim=1).view(batch_size, -1)
        fcs, fcvars = [], []
        for i in range(self.num_class):
            fcs.append(self.fcs[i](output))
            fcvars.append(self.fcvars[i](output))
        return fcs, fcvars

