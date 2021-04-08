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
import torch.nn.functional as F
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
        num_workers=2,
        pin_memory=True
        )
    
    valid_source_loader = torch.utils.data.DataLoader(
        dataset=valid_source_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
        )
    
    valid_target_loader = torch.utils.data.DataLoader(
        dataset=valid_target_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
        )

    dataloaders_dict = {"train": train_loader, "valid_source": valid_source_loader, "valid_target": valid_target_loader}
    
    return dataloaders_dict

#############################################################################
# training
#############################################################################
def calc_auc(y_true, y_pred):
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config["etc"]["max_fpr"])
    return auc, p_auc

# extract function
def train_net(net, dataloaders_dict, optimizer, num_epochs, writer, model_out_path):
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    
    output_dicts = {}
    
    for epoch in range(num_epochs):
        best_val_losses = np.inf
        best_epoch = 0
        best_flag = False
        for phase in ['train', 'valid_source', 'valid_target']:

            tr_losses = 0
            val_losses = 0
            
            if phase == 'train':
                net.train()
                
                for sample in tqdm(dataloaders_dict[phase]):
                    # feature
                    input = sample['feature']
                    input = input.to(device)
                    # target
                    section_type = sample['type']
                    section_type = section_type.to(device)
                    # model
                    output_dict = net(input, section_type, device)
                    # calc loss
                    loss = F.cross_entropy(output_dict['pred_section_type'], section_type)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tr_losses += loss.item()
                tr_losses = tr_losses / len(dataloaders_dict[phase])
                # tensorboard
                writer.add_scalar("loss", tr_losses, epoch+1)

            else:
                net.eval()
                preds = np.zeros(len(dataloaders_dict[phase].dataset))
                labels = np.zeros(len(dataloaders_dict[phase].dataset))
                for sample in tqdm(dataloaders_dict[phase]):
                    # feature
                    input = sample['feature']
                    input = input.to(device)
                    # target
                    section_type = sample['type']
                    section_type = section_type.to(device)
                    with torch.no_grad():
                        # model
                        output_dict = net(input, section_type, device)
                        # calc loss
                        loss = F.cross_entropy(output_dict['pred_section_type'], section_type)
                        val_losses += loss.item()

                        #labels[idx] = label.item()
                        #preds[idx] = loss.to('cpu').detach().numpy().copy()
                # calc epoch score
                val_losses = val_losses / len(dataloaders_dict[phase])
                #val_AUC, val_pAUC = calc_auc(labels, preds)
                
                if phase == 'valid_source':
                    writer.add_scalar("val_source_loss", val_losses, epoch+1)
                    val_losses = val_losses.to('cpu').detach().numpy().copy()
                    if best_val_losses > val_losses:
                        best_val_losses = val_losses.to('cpu').detach().numpy().copy()
                        best_epoch = epoch.copy()
                        best_flag = True
                        # save
                        torch.save(best_model.state_dict(), model_out_path)
                        logger.info("Save best model")
                    val_source_losses = val_losses.copy()
                    #writer.add_scalar("val_source_AUC", val_AUC, epoch+1)
                    #writer.add_scalar("val_source_pAUC", val_pAUC, epoch+1)
                else:
                    writer.add_scalar("val_target_loss", val_losses, epoch+1)
                    val_target_losses = val_losses.copy()
                    #writer.add_scalar("val_target_AUC", val_AUC, epoch+1)
                    #writer.add_scalar("val_target_pAUC", val_pAUC, epoch+1)
    logger.info(f"{epoch}/{num_epochs} \
                train_losses : {tr_losses}, \
                val_source_losses : {val_source_losses}, \
                val_target_losses : {val_target_losses}, \
                best_flag : {best_flag}")
    
    output_dicts = {'best_epoch':best_epoch, 'best_val_losses':best_val_losses}
    return output_dicts

# extract function
def extract_net(net, dataloaders_dict):
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    
    output_dicts = {}
    
    for phase in ['train', 'valid_source', 'valid_target']:
        net.eval()
        M_means = []
        labels = []
        wav_names = []
        for sample in tqdm(dataloaders_dict[phase]):
            wav_name = sample['wav_name']
            wav_names = wav_names + wav_name
            input = sample['feature']
            input = input.to(device)
            label = sample['label'].to('cpu')
            labels.append(label)

            with torch.no_grad():
                output_dict = net(input, device)  # (batch_size,input(2D)) 
                M_means.append(output_dict['M_means'].to('cpu'))
                
        M_means = torch.cat(M_means, dim=0).detach().numpy().copy()
        labels = torch.cat(labels, dim=0).detach().numpy().copy()
        output_dicts[phase] = {'features' : M_means, 'labels' : labels, 'wav_names' : wav_names}
    
    return output_dicts