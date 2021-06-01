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
        shuffle=False,
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
    #logger.info("AUC : {}".format(auc))
    #logger.info("pAUC : {}".format(p_auc))
    return auc, p_auc

# training function
def extract_net(net, dataloaders_dict):
    outputs = []
    def hook(module, input, output):
        #print(output.shape)
        output = output.cpu()
        outputs.append(output.mean(dim=(2,3)))
    
    for i in range(len(net.resnet.layer1)):
        net.resnet.layer1[i].register_forward_hook(hook)
    for i in range(len(net.resnet.layer2)):
        net.resnet.layer2[i].register_forward_hook(hook)
    for i in range(len(net.resnet.layer3)):
        net.resnet.layer3[i].register_forward_hook(hook)
    for i in range(len(net.resnet.layer4)):
        net.resnet.layer4[i].register_forward_hook(hook)
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
                _ = net(input, device)  # (batch_size,input(2D))
                outputs = torch.cat(outputs, dim=1).cpu()
                M_means.append(outputs)
                outputs = []
                
        M_means = torch.cat(M_means, dim=0).detach().numpy().copy()
        labels = torch.cat(labels, dim=0).detach().numpy().copy()
        output_dicts[phase] = {'features' : M_means, 'labels' : labels, 'wav_names' : wav_names}
    
    return output_dicts