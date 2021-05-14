import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torchvision import models
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

def zscore(x, dim = None):
    xmean = x.mean(dim=dim, keepdims=True)
    xstd  = torch.std(x, dim=dim, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

class FC_block(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(FC_block, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.layer_norm1 = nn.LayerNorm(out_features)
    
    def forward(self, input, use_tanh=False):
        if use_tanh == True:
            x = torch.tanh_(self.layer_norm1(self.fc1(input)))
        else:
            x = F.relu_(self.layer_norm1(self.fc1(input)))
        return x

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Variational_Inference(nn.Module):
    def __init__(self, in_features, out_features):
        self.eps = 1e-12
        super(Variational_Inference, self).__init__()
        
        self.fc_mean = nn.Linear(in_features, out_features) 
        self.fc_var = nn.Linear(in_features, out_features)
    
    def forward(self, input, device='cuda:0'):
        
        x = input
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        KLd = -0.5 * torch.sum(1 + var - mean**2 - torch.exp(var+self.eps), dim=(0,2))
        #print(KLd.shape)
        z = self.sample_z(mean, var, device)
        
        return z, KLd
    
    def sample_z(self, mean, var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5*var + self.eps)

class Decoder_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_ratio=0.1):
        super(Decoder_Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_ratio)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = PositionwiseFeedForward(d_model=embed_dim, hidden=embed_dim*2, drop_prob=0.1)
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x, mask=None):
        x_ = x.clone()
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=mask)
        # layer norm + add
        x = self.layer_norm(attn_output + x_)
        x_ = x.clone()
        x = self.ffn(x)
        x = x + x_
        #x = self.layer_norm2(x)
        # dropout
        x = self.dropout(x)
        return x, attn_weights

class Transformer_AE(nn.Module):

    def __init__(self, embed_dim, seq_len):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        # model architecture
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1, max_len=self.seq_len)
        self.decoder_block1 = Decoder_Block(embed_dim, num_heads=4, dropout_ratio=0.1)
        self.decoder_block2 = Decoder_Block(embed_dim, num_heads=4, dropout_ratio=0.1)
        self.decoder_block3 = Decoder_Block(embed_dim, num_heads=4, dropout_ratio=0.1)
        self.fc_block1 = FC_block(embed_dim, int(embed_dim/2))
        self.fc_block2 = FC_block(int(embed_dim/2), embed_dim)
        self.decoder_block4 = Decoder_Block(embed_dim, num_heads=4, dropout_ratio=0.1)
        self.decoder_block5 = Decoder_Block(embed_dim, num_heads=4, dropout_ratio=0.1)
        self.decoder_block6 = Decoder_Block(embed_dim, num_heads=4, dropout_ratio=0.1)
        #self.valiational_infe = Variational_Inference(embed_dim, embed_dim)
        #self.fc_block1 = FC_block(embed_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        # shape: (63, batch, 128)
        seq_len = inputs.shape[0]
        #mask = self.generate_square_subsequent_mask(seq_len)
        #mask = mask.cuda()
        #print('mask',mask.shape)
        #print('inputs',inputs.shape)
        outputs = self.pos_encoder(inputs)
        outputs, _ = self.decoder_block1(inputs)
        outputs, _ = self.decoder_block2(outputs)
        outputs, attn_weights = self.decoder_block3(outputs)
        outputs = self.fc_block1(outputs)
        outputs = self.fc_block2(outputs)
        outputs, attn_weights = self.decoder_block4(outputs)
        outputs, attn_weights = self.decoder_block5(outputs)
        outputs, attn_weights = self.decoder_block6(outputs)
        #outputs, KLd = self.valiational_infe(outputs)
        #outputs = self.fc_block1(outputs)
        
        outputs = self.fc1(outputs)
    
        return outputs, attn_weights

    def generate_square_subsequent_mask(self, sz):
        r"""
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def get_loss(self, inputs, labels=None, imshow=False):
        #print(inputs.shape)
        # inputs : [batch, 64, 128] -> [64, batch, 128] (time, batch, melbins)
        inputs = inputs.permute(1,0,2)
        hat_inputs, attn_weights = self.forward(inputs)
        if self.training == True:
            loss = self.criterion(hat_inputs, inputs)
        else:
            loss = F.mse_loss(hat_inputs, inputs, reduce=False)
            #print(loss)
        #loss = self.criterion(output, labels)
        if imshow == True:
            plt.imshow(inputs[:,0,:].squeeze(1).cpu().detach().T, aspect='auto')
            plt.show()
            plt.imshow(hat_inputs[:,0,:].squeeze(1).cpu().detach().T, aspect='auto')
            plt.show()
            plt.imshow(attn_weights[0,:,:].squeeze(0).cpu().detach(), aspect='auto')
            plt.show()
        return loss#, hat_inputs.permute()
