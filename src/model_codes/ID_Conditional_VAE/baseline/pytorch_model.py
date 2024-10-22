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
        x = F.relu_(self.bn1(self.fc1(x)))
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
    def __init__(self, in_features, mid_size, latent_size):
        
        super(Conditional_VAE, self).__init__()

        self.in_features = in_features
        
        self.bn0 = nn.BatchNorm1d(in_features)
        self.Encoder = Encoder(in_features, mid_size)
        
        self.fc1 = nn.Linear(mid_size, latent_size)
        self.bn1 = nn.BatchNorm1d(latent_size)
        self.fc2 = nn.Linear(latent_size, mid_size)
        self.bn2 = nn.BatchNorm1d(mid_size)
        
        self.Decoder = Decoder(mid_size, in_features)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        
        x = self.bn0(input)
        x = self.Encoder(x)
        
        x = F.relu_(self.bn1(self.fc1(x)))
        x = F.relu_(self.bn2(self.fc2(x)))
        
        x = self.Decoder(x)

        if self.training == True:
            reconst_error = F.mse_loss(x, input, reduction='mean')
        else:
            reconst_error = F.mse_loss(x, input, reduction='none').mean(dim=1)
            
        output_dict = {'reconst_error': reconst_error, 'reconstruction': x}
        
        return output_dict
