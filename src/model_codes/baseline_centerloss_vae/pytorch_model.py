"""
Class definition of AutoEncoder in PyTorch.

Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #self.ln1 = nn.LayerNorm(out_features)
    
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
        self.eps = 1e-12
        super(Bottleneck, self).__init__()
        
        self.fc_mean = nn.Linear(in_features, out_features) 
        self.fc_var = nn.Linear(in_features, out_features)
    
    def forward(self, input, device='cuda:0'):
        
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

class AutoEncoder(nn.Module):
    """
    AutoEncoder
    """

    def __init__(self, x_dim, h_dim, z_dim, n_hidden):
        super(AutoEncoder, self).__init__()

        in_features = x_dim
        mid_size = h_dim
        latent_size = z_dim
        
        self.n_hidden = n_hidden  # number of hidden layers
        self.Encoder = Encoder(in_features, mid_size)
        self.fc_block1 = FC_block(mid_size, latent_size)
        
        self.Bottleneck = Bottleneck(latent_size, latent_size)
        
        self.fc_block2 = FC_block(latent_size, mid_size)
        self.Decoder = Decoder(mid_size, in_features)
        self.out_fc = nn.Linear(in_features, in_features)
        
        # for center_loss
        self.latent_size = latent_size
        self.num_classes = 6
        self.center_weight = 150
        #self.out_classes = nn.Linear(latent_size, num_classes)
        self.center_loss = CenterLoss(num_classes=self.num_classes, feat_dim=latent_size, use_gpu=True)

        #layers = nn.ModuleList([])
        #layers += [nn.Linear(x_dim, h_dim)]
        #layers += [nn.Linear(h_dim, h_dim) for _ in range(self.n_hidden)]
        #layers += [nn.Linear(h_dim, z_dim)]
        #layers += [nn.Linear(z_dim, h_dim)]
        #layers += [nn.Linear(h_dim, h_dim) for _ in range(self.n_hidden)]
        #layers += [nn.Linear(h_dim, x_dim)]
        #self.layers = nn.ModuleList(layers)

        #bnorms = [nn.BatchNorm1d(h_dim) for _ in range(self.n_hidden + 1)]
        #bnorms += [nn.BatchNorm1d(z_dim)]
        #bnorms += [nn.BatchNorm1d(h_dim) for _ in range(self.n_hidden + 1)]
        #self.bnorms = nn.ModuleList(bnorms)

        #self.activation = nn.ReLU()
        #self.criterion = nn.MSELoss()

    def forward(self, inputs, label):
        """
        Reconstruct inputs through AutoEncoder.
        """
        x = self.Encoder(inputs)
        x = self.fc_block1(x, use_tanh=False)
        center_loss = self.center_loss(x, label)
        z, KLd = self.Bottleneck(x)
        x = self.fc_block2(z)
        output = self.Decoder(x)
        output = self.out_fc(output)
        

        return output, KLd, center_loss

    def get_loss(self, inputs, label):
        """
        Calculate loss function of AutoEncoder.
        """
        recon_x, KLd, center_loss = self.forward(inputs, label)
        if self.training == True:
            recon_loss = F.mse_loss(recon_x, inputs, reduction='mean')
            recon_loss = recon_loss + KLd.mean(dim=0) + center_loss*self.center_weight
        else:
            recon_loss = F.mse_loss(recon_x, inputs, reduction='none').mean(dim=1)
            recon_loss = recon_loss + KLd + center_loss

        return recon_loss
