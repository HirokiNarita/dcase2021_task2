import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

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

class Decoder_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_ratio=0.1):
        super(Decoder_Block, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_ratio)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x, mask):
        # layer norm
        x = self.layer_norm1(x)
        #print(x)
        # self attention
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=mask)
        # layer norm -> linear
        attn_output = self.fc1(self.layer_norm2(attn_output))
        # skip-connection -> ReLU
        x = F.relu_(x + attn_output)
        # dropout
        x = self.dropout(x)
        return x, attn_weights

class Transformer_Decoder(nn.Module):

    def __init__(self, embed_dim, seq_len):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        # model architecture
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1, max_len=128)
        self.decoder_block1 = Decoder_Block(embed_dim, num_heads=8, dropout_ratio=0.1)
        self.decoder_block2 = Decoder_Block(embed_dim, num_heads=8, dropout_ratio=0.1)
        self.decoder_block3 = Decoder_Block(embed_dim, num_heads=8, dropout_ratio=0.1)
        self.decoder_block4 = Decoder_Block(embed_dim, num_heads=8, dropout_ratio=0.1)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        # shape: (63, batch, 128)
        seq_len = inputs.shape[0]
        mask = self.generate_square_subsequent_mask(seq_len)
        mask = mask.cuda()
        #print('mask',mask.shape)
        #print('inputs',inputs.shape)
        outputs = self.pos_encoder(inputs)
        outputs, _ = self.decoder_block1(outputs, mask)
        outputs, attn_weights = self.decoder_block2(outputs, mask)
        #outputs = self.decoder_block3(outputs, mask)
        #outputs = self.decoder_block4(outputs, mask)
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
        # stride
        # t0,t1,t2... -> t1,t2,t3...
        # stride
        gt_inputs = inputs[1:, :, :] # [63, batch, 128] 1~64
        inputs = inputs[:63, :, :] # [63, batch, 128] 0~63
        #print(inputs.shape)
        hat_inputs, attn_weights = self.forward(inputs)
        if self.training == True:
            loss = self.criterion(hat_inputs, gt_inputs)
        else:
            loss = torch.mean(torch.square_(gt_inputs - hat_inputs), dim=(0,2))
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
