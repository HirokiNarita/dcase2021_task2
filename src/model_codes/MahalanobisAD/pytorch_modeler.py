############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime

# general analysis tool-kit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# pytorch
import torch
import torch.utils.data as data
from torch import optim, nn
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter

# deeplearning tool-kit
from torchvision import transforms

# etc
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from tqdm import tqdm
from collections import defaultdict

# original library
import common as com
import preprocessing as prep

############################################################################
# load config
############################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)
log_folder = config['IO_OPTION']['OUTPUT_ROOT']+'/{0}.log'.format(datetime.date.today())
logger = com.setup_logger(log_folder, 'pytorch_modeler.py')
############################################################################
# Setting seed
############################################################################
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

############################################################################
# Make Dataloader
############################################################################
def make_dataloader(train_paths, machine_type):
    transform = transforms.Compose([
        prep.extract_waveform(),
        prep.ToTensor()
    ])
    train_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['train'], transform=transform)
    valid_source_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_source'], transform=transform)
    valid_target_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_target'], transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=config['param']['shuffle'],
        )
    
    valid_source_loader = torch.utils.data.DataLoader(
        dataset=valid_source_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        )
    
    valid_target_loader = torch.utils.data.DataLoader(
        dataset=valid_target_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        )

    dataloaders_dict = {"train": train_loader, "valid_source": valid_source_loader, "valid_target": valid_target_loader}
    
    return dataloaders_dict

#############################################################################
# training
#############################################################################
def calc_auc(y_true, y_pred):
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config["etc"]["max_fpr"])
    #logger.info("AUC : {}".format(auc))
    #logger.info("pAUC : {}".format(p_auc))
    return auc, p_auc

# training function
def train_net(net, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, writer):
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)

    tr_epoch_losses = []
    #reconstruct_img = defaultdict(list)
    #epoch_valid_score = defaultdict(list)
    # epochループ開始
    for epoch in range(num_epochs):
        # loss
        tr_losses = 0
        #losses['valid'] = 0
        labels = []

        # epochごとの訓練と検証のループ
        for phase in ['train', 'valid_source', 'valid_target']:
            if phase == 'train':
                net.eval()
                M_means = []
                labels = []
                for sample in tqdm(dataloaders_dict[phase]):
                    input = sample['feature']
                    input = input.to(device)
                    label = sample['label'].to('cpu')
                    labels.append(label)
                    # optimizerを初期化
                    optimizer.zero_grad()

                    with torch.no_grad():
                        output_dict = net(input, device)  # (batch_size,input(2D)) 
                        M_means.append(output_dict['M_means'].to('cpu'))
                        
                M_means = (torch.cat(M_means, dim=0) + 1).log()
                #labels = torch.cat(labels, dim=0)
                #labels = labels.unsqueeze(-1)
                
                plt.imshow(M_means, aspect='auto', cmap='jet')
                plt.title(phase)
                plt.colorbar()
                plt.show()
                #tr_epoch_losses.append(tr_losses / len(dataloaders_dict['train']))
                # tb
                #writer.add_scalar("loss", tr_epoch_losses[-1], epoch+1)
            else:
                preds = np.zeros(len(dataloaders_dict[phase].dataset))
                labels = np.zeros(len(dataloaders_dict[phase].dataset))
                net.eval()
                M_means = []
                labels = []
                for sample in tqdm(dataloaders_dict[phase]):
                    input = sample['feature']
                    input = input.to(device)
                    label = sample['label'].to('cpu')
                    labels.append(label)
                    # optimizerを初期化
                    optimizer.zero_grad()

                    with torch.no_grad():
                        output_dict = net(input, device)  # (batch_size,input(2D)) 
                        M_means.append(output_dict['M_means'].to('cpu'))
                
                M_means = (torch.cat(M_means, dim=0) + 1).log()
                #labels = torch.cat(labels, dim=0)
                #labels = labels.unsqueeze(-1)
                
                print(M_means.shape)
                plt.imshow(M_means, aspect='auto', cmap='jet')
                plt.title(phase)
                plt.colorbar()
                plt.show()
                """
                valid_AUC, valid_pAUC = calc_auc(labels, preds)
                epoch_valid_score['AUC'].append(valid_AUC)
                epoch_valid_score['pAUC'].append(valid_pAUC)

                writer.add_scalar("valid_AUC", valid_AUC, epoch+1)
                writer.add_scalar("pAUC", valid_pAUC, epoch+1)
                
                if ((epoch+1) % 10 == 0) or (epoch == 0):
                    plt.imshow(y[0,:,:].to('cpu').detach().numpy(), aspect='auto')
                    plt.show()
                    reconstruct_img['input'].append(x)
                    reconstruct_img['output'].append(y)
                    reconstruct_img['label'].append(label)
        
            # データローダーからminibatchを取り出すループ
        #epoch_losses['valid'].append(losses['valid'] / len(dataloaders_dict['valid']))
        
        logger.info('Epoch {}/{}:train_loss:{:.6f}, valid_AUC:{:.6f}, valid_pAUC:{:.6f}'.format(epoch+1,
                                                                                                num_epochs,
                                                                                                tr_epoch_losses[-1],
                                                                                                epoch_valid_score['AUC'][-1],
                                                                                                epoch_valid_score['pAUC'][-1]))
        """
    return output_dict
    #return {'train_epoch_score':tr_epoch_losses, 'valid_epoch_score':epoch_valid_score, 'reconstruct_img':reconstruct_img, 'model':net}
