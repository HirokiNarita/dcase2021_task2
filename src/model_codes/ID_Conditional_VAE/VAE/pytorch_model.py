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
    
    def forward(self, input, use_tanh=False):
        x = input
        if use_tanh == True:
            x = torch.tanh_(self.bn1(self.fc1(x)))
        else:
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

class Bottleneck(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(Bottleneck, self).__init__()
        
        self.fc_mean = nn.Linear(in_features, out_features) 
        self.fc_var = nn.Linear(in_features, out_features)
    
    def forward(self, input, device):
        
        x = input
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        
        KLd = -0.5 * torch.sum(1 + var - mean**2 - torch.exp(var), dim=1)
        #print(KLd.shape)
        z = self.sample_z(mean, var, device)
        
        return z, KLd
    
    def sample_z(self, mean, var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5*var)

class Conditional_VAE(nn.Module):
    def __init__(self, in_features, mid_size, latent_size):
        
        super(Conditional_VAE, self).__init__()

        self.in_features = in_features
        
        self.bn0 = nn.BatchNorm1d(in_features)
        self.Encoder = Encoder(in_features, mid_size)
        self.fc_block1 = FC_block(mid_size, latent_size)
        
        self.Bottleneck = Bottleneck(latent_size, latent_size)
        
        self.fc_block2 = FC_block(latent_size, mid_size)
        self.Decoder = Decoder(mid_size, in_features)

    def forward(self, input, device):
        """
        Input: (batch_size, data_length)"""
        
        x = self.bn0(input)
        #gt_x = x.clone()
        x = self.Encoder(x)
        x = self.fc_block1(x, use_tanh=True)
        
        z, KLd = self.Bottleneck(x, device)
        #print(KLd)
        x = self.fc_block2(z)
        x = self.Decoder(x)
        #plt.plot(KLd.to('cpu').detach().numpy())
        #plt.show()
        if self.training == True:
            reconst_error = F.mse_loss(x, input, reduction='mean')
            reconst_error = reconst_error + KLd.mean(dim=0)
        else:
            reconst_error = F.mse_loss(x, input, reduction='none').mean(dim=1)
            #reconst_error = reconst_error + KLd
            #print(reconst_error)
            
        output_dict = {'reconst_error': reconst_error, 'reconstruction': x}
        
        return output_dict
