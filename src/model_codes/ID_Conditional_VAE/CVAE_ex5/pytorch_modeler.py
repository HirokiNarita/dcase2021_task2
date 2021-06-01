############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime
from copy import copy

# general analysis tool-kit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from IPython.display import display
from sklearn.preprocessing import StandardScaler
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
log_file = config['IO_OPTION']['OUTPUT_ROOT']+'/train_{0}.log'.format(datetime.date.today())
logger = com.setup_logger(log_file, 'pytorch_modeler.py')
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
def make_dataloader(ext_data):
    # zscore scaling
    #scaler = StandardScaler()
    #ext_data['train']['features'] = scaler.fit_transform(ext_data['train']['features'])
    #ext_data['valid_source']['features'] = scaler.fit_transform(ext_data['valid_source']['features'])
    #ext_data['valid_target']['features'] = scaler.fit_transform(ext_data['valid_target']['features'])
    
    train_dataset = prep.DCASE_task2_Dataset(ext_data['train'])
    valid_source_dataset = prep.DCASE_task2_Dataset(ext_data['valid_source'])
    valid_target_dataset = prep.DCASE_task2_Dataset(ext_data['valid_target'])

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

# extract function
def train_net(net, dataloaders_dict, optimizer, num_epochs, writer, model_out_path, score_out_path, pred_out_path):
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    
    output_dicts = {}
    best_criterion = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        best_flag = False
        all_scores_df = pd.DataFrame()
        for phase in ['train', 'valid_source', 'valid_target']:
            logger.info(phase)
            if phase == 'train':
                net.train()
                section_types = []
                tr_losses = 0
                for sample in tqdm(dataloaders_dict[phase]):
                    # feature
                    input = sample['features']
                    input = input.to(device)
                    section_type = sample['type']
                    section_type = section_type.to(device)
                    target_bool = sample['target_bool']
                    target_bool = target_bool.to(device)
                    # model
                    output_dict = net(input, section_type, target_bool, device)
                    # calc loss
                    loss, x_hat = output_dict['reconst_error'], output_dict['reconstruction']
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()
                    tr_losses += loss.item()
                tr_losses = tr_losses / len(dataloaders_dict[phase])
                # tensorboard
                writer.add_scalar("tr_loss", tr_losses, epoch+1)

            elif phase == 'valid_source':
                net.eval()
                preds = []
                labels = []
                section_types = []
                
                inputs = []
                x_hats = []
                for sample in tqdm(dataloaders_dict[phase]):
                    # feature
                    input = sample['features']
                    input = input.to(device)
                    section_type = sample['type']
                    section_type = section_type.to(device)
                    target_bool = sample['target_bool']
                    target_bool = target_bool.to(device)
                    with torch.no_grad():
                        # model
                        output_dict = net(input, section_type, target_bool, device)
                        # calc loss
                        reconst_error, x_hat = output_dict['reconst_error'], output_dict['reconstruction']
                        # to numpy
                        reconst_error = reconst_error.to('cpu')
                        #x_hat = x_hat.to('cpu')
                        # append
                        #inputs.append(input.to('cpu'))
                        x_hats.append(x_hat.to('cpu'))
                        preds.append(reconst_error)
                        labels.append(sample['label'])
                        section_types.append(sample['type'])
                # concat
                
                preds = torch.cat(preds).detach().numpy().copy()
                labels = torch.cat(labels).detach().numpy().copy()
                x_hats = torch.cat(x_hats).detach().numpy().copy()
                section_types = torch.cat(section_types).detach().numpy().copy()
                # calc score
                all_scores_df = com.calc_DCASE2021_score(all_scores_df, labels, preds, section_types, phase)
                # preds
                describe_df = com.get_pred_discribe(labels, preds, section_types)
            else:
                net.eval()
                preds = []
                labels = []
                section_types = []
                
                x_hats = []
                for sample in tqdm(dataloaders_dict[phase]):
                    # feature
                    input = sample['features']
                    input = input.to(device)
                    section_type = sample['type']
                    section_type = section_type.to(device)
                    target_bool = sample['target_bool']
                    target_bool = target_bool.to(device)
                    with torch.no_grad():
                        # model
                        output_dict = net(input, section_type, target_bool, device)
                        # calc loss
                        reconst_error, x_hat = output_dict['reconst_error'], output_dict['reconstruction']
                        # to numpy
                        reconst_error = reconst_error.to('cpu')
                        #x_hat = x_hat.to('cpu')
                        # append
                        x_hats.append(x_hat.to('cpu'))
                        preds.append(reconst_error)
                        labels.append(sample['label'])
                        section_types.append(sample['type'])

                # concat
                x_hats = torch.cat(x_hats).detach().numpy().copy()
                preds = torch.cat(preds).detach().numpy().copy()
                labels = torch.cat(labels).detach().numpy().copy()
                section_types = torch.cat(section_types).detach().numpy().copy()
                # calc score
                all_scores_df = com.calc_DCASE2021_score(all_scores_df, labels, preds, section_types, phase)
                # preds
                describe_df_ = com.get_pred_discribe(labels, preds, section_types)
                describe_df = describe_df.append(describe_df_)
        #display(all_scores_df)
        # get hmean
        val_AUC_hmean = all_scores_df.loc['h_mean']['AUC']
        val_pAUC_hmean = all_scores_df.loc['h_mean']['pAUC']
        # tensorboard
        writer.add_scalar("val_AUC_hmean", val_AUC_hmean, epoch+1)
        writer.add_scalar("val_pAUC_hmean", val_pAUC_hmean, epoch+1)
        # early stopping
        if best_criterion < val_pAUC_hmean:
            best_score = all_scores_df.copy()
            best_tr_losses = tr_losses
            best_criterion = val_pAUC_hmean.copy()
            best_pred = describe_df.copy()
            best_epoch = epoch
            best_model = net
            best_flag = True

            plt.imshow(x_hats, aspect='auto')
            plt.show()
            # save
            torch.save(best_model.state_dict(), model_out_path)
            best_score.to_csv(score_out_path)
            best_pred.to_csv(pred_out_path)
            # display score dataframe
            display(best_score)
            #logger.info("Save best model")
        # logger info
        epoch_log = (
                    f"epoch:{epoch+1}/{num_epochs},"
                    f" train_losses:{tr_losses:.6f},"
                    f" val_AUC_hmean:{val_AUC_hmean:.6f},"
                    f" val_pAUC_hmean:{val_pAUC_hmean:.6f},"
                    f" best_flag:{best_flag}"
                    )
        logger.info(epoch_log)
    # display best score
    best_log = (
                f"best model,"
                f" epoch:{best_epoch+1}/{num_epochs},"
                f" train_losses:{best_tr_losses:.6f},"
                f" val_pAUC_hmean:{best_criterion:.6f},"
                )
    logger.info(best_log)
    display(best_score)
    output_dicts = {'best_epoch':best_epoch, 'best_pAUC':best_criterion, 'best_pred':best_pred}
    return output_dicts

"""
# inference function
def inference_net(net, dataloaders_dict):
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    
    output_dicts = {}
    best_val_losses = np.inf
    best_epoch = 0
    
    for phase in ['valid_source', 'valid_target']:
        logger.info(phase)

        if phase == 'valid_source':
            net.eval()
            preds = np.zeros(len(dataloaders_dict[phase].dataset))
            labels = np.zeros(len(dataloaders_dict[phase].dataset))
            for sample in tqdm(dataloaders_dict[phase]):
                # feature
                input = sample['features']
                input = input.to(device)
                # target
                section_type = sample['type']
                section_type = section_type.to(device)
                with torch.no_grad():
                    # model
                    output_dict = net(input, section_type, mixup_lambda=None, layer_out=False)
                    pred = output_dict['pred_section_type']
        else:
            net.eval()
            preds = np.zeros(len(dataloaders_dict[phase].dataset))
            labels = np.zeros(len(dataloaders_dict[phase].dataset))
            for idx, sample in enumerate(tqdm(dataloaders_dict[phase])):
                # feature
                input = sample['features']
                input = input.to(device)
                # target
                section_type = sample['type']
                section_type = section_type.to(device)
                with torch.no_grad():
                    # model
                    output_dict = net(input, mixup_lambda=None, layer_out=False)
                    pred = output_dict['pred_section_type']
                    preds[idx+1 * sample]

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
                output_dict = net(input, layer_out=True)  # (batch_size,input(2D)) 
                M_means.append(output_dict['M_means'].to('cpu'))
                
        M_means = torch.cat(M_means, dim=0).detach().numpy().copy()
        labels = torch.cat(labels, dim=0).detach().numpy().copy()
        output_dicts[phase] = {'features' : M_means, 'labels' : labels, 'wav_names' : wav_names}
    
    return output_dicts

"""