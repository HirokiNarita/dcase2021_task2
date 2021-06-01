import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from pytorch_utils import do_mixup, interpolate, pad_framewise_output
import matplotlib.pyplot as plt
#output_dict = {'loss':loss, 'x':input_spec, 'y':y}

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition.
    ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        #print(x.shape)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size,
                                                                  self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        #print('distmat', distmat.shape)
        #print(x.shape, self.centers.t().shape)
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #print(distmat.shape)
        dist = distmat * mask.float()
        if self.training == True:
            loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        else:
            prob = dist.clamp(min=1e-12, max=1e+12)
            loss = prob.mean(dim=1)

        return loss



class FC_block(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(FC_block, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.prelu = nn.PReLU(out_features)
        #self.ln1 = nn.LayerNorm(out_features)
    
    def forward(self, input, use_silu=False):
        x = input
        if use_silu == True:
            x = self.prelu(self.bn1(self.fc1(x)))
        else:
            x = F.relu_(self.bn1(self.fc1(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(Encoder, self).__init__()
        
        self.fc_block1 = FC_block(in_features, out_features)
        self.fc_block2 = FC_block(out_features, out_features)
        self.fc_block3 = FC_block(out_features, out_features)
    
    def forward(self, input, use_silu=False):
        x = input
        x = self.fc_block1(x, use_silu)
        x = self.fc_block2(x, use_silu)
        x = self.fc_block3(x, use_silu)
        return x

class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(Decoder, self).__init__()
        
        self.fc_block1 = FC_block(in_features, out_features)
        self.fc_block2 = FC_block(out_features, out_features)
        self.fc_block3 = FC_block(out_features, out_features)
    
    def forward(self, input, use_silu=False):
        x = input
        x = self.fc_block1(x, use_silu)
        x = self.fc_block2(x, use_silu)
        x = self.fc_block3(x, use_silu)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_features, out_features):
        self.eps = 1e-12
        super(Bottleneck, self).__init__()
        
        self.fc_mean = nn.Linear(in_features, out_features) 
        self.fc_var = nn.Linear(in_features, out_features)
    
    def forward(self, input, device):
        
        x = input
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        
        KLd = -0.5 * torch.sum(1 + var - mean**2 - torch.exp(var+self.eps), dim=1)
        #print(KLd.shape)
        z = self.sample_z(mean, var, device)
        
        return z, KLd
    
    def sample_z(self, mean, var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5*var + self.eps)

class Conditional_VAE(nn.Module):
    def __init__(self, in_features, mid_size, latent_size, num_classes):
        
        super(Conditional_VAE, self).__init__()
        
        #self.bn0 = nn.BatchNorm1d(in_features)
        self.Encoder = Encoder(in_features, mid_size)
        self.fc_block1 = FC_block(mid_size, latent_size)
        
        self.Bottleneck = Bottleneck(latent_size, latent_size)
        
        self.fc_block2 = FC_block(latent_size, mid_size)
        self.Decoder = Decoder(mid_size, in_features)
        self.out_fc = nn.Linear(in_features, in_features)

        self.kld_beta = 1
        
        # for center_loss
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.center_weight = 1
        #self.out_classes = nn.Linear(latent_size, num_classes)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=latent_size, use_gpu=True)
        
    def forward(self, input, section_type, target_bool, device):
        x = input
        #x = self.bn0(input)
        x_gt = x.clone()
        x = self.Encoder(x, True)
        x = self.fc_block1(x, True)
        # for center_loss
        #out_classes = self.out_classes(x)
        #print(x.shape)
        #print(self.latent_size)
        center_loss = self.center_loss(x, section_type)
        #section_type = F.one_hot(section_type, num_classes=self.num_classes)
        
        z, KLd = self.Bottleneck(x, device)
        # to one-hot
        #if self.training == True:
            #section_type = F.one_hot(section_type, num_classes=self.num_classes)
        #    target_bool = F.one_hot(target_bool, num_classes=2)
        #else:
            #section_type = F.one_hot(section_type, num_classes=self.num_classes)
        #    target_bool = F.one_hot(target_bool, num_classes=2)
            #section_type = torch.full((x.shape[0], 1), 0).squeeze(1).to(device)
            #section_type = F.one_hot(section_type, num_classes=self.num_classes)
        
        #section_type = F.one_hot(section_type, num_classes=self.num_classes)
        # concat
        #z = torch.cat([x, section_type], dim=1)
        #target_bool = target_bool.unsqueeze(1)
        #z = torch.cat([z, target_bool], dim=1)
        
        x = self.fc_block2(z, True)
        x = self.Decoder(x, True)
        x = self.out_fc(x)

        if self.training == True:
            reconst_error = F.mse_loss(x, x_gt, reduction='mean')
            #reconst_error = -(F.cosine_similarity(x, x_gt, dim=1)).mean()
            reconst_error = reconst_error + KLd.mean(dim=0)*self.kld_beta + center_loss*self.center_weight
        else:
            #reconst_error = -F.cosine_similarity(x, x_gt, dim=1)
            reconst_error = F.mse_loss(x, x_gt, reduction='none').mean(dim=1)
            reconst_error = reconst_error + KLd + center_loss
            #print(reconst_error.shape)
            
        output_dict = {'reconst_error': reconst_error, 'reconstruction': x, 'KLd': KLd.mean(), 'center_loss': center_loss.mean()}
        
        return output_dict
