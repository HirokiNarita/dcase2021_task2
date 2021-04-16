import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from pytorch_utils import do_mixup, interpolate, pad_framewise_output
import matplotlib.pyplot as plt
#output_dict = {'loss':loss, 'x':input_spec, 'y':y}

class FC_block(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(FC_block, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
    
    def forward(self, input):
        x = input
        x = F.relu_(self.bn1(self.fc1))
        return x

class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(Encoder, self).__init__()
        
        self.fc_block1 = FC_block(in_features, out_features)
        self.fc_block2 = FC_block(out_features, out_features)
        self.fc_block3 = FC_block(out_features, out_features)
    
    def forward(self, input):
        x = input
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(Decoder, self).__init__()
        
        self.fc_block1 = FC_block(in_features, out_features)
        self.fc_block2 = FC_block(out_features, out_features)
        self.fc_block3 = FC_block(out_features, out_features)
    
    def forward(self, input):
        x = input
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        return x


class Conditional_VAE(nn.Module):
    def __init__(self, in_features, latent_size):
        
        super(Conditional_VAE, self).__init__()

        self.in_features = in_features
        
        self.bn0 = nn.BatchNorm2d(in_features)
        self.Encoder = Encoder(in_features, latent_size)
        
        self.Decoder = Decoder(latent_size, in_features)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        
        x = self.bn0(input)
        x = self.Encoder(x)
        x = self.Decoder(x)
        
        reconst_loss = F.mse_loss(x, input)
        output_dict = {'reconst_loss': reconst_loss, 'reconstruction': x}
        
        return output_dict
