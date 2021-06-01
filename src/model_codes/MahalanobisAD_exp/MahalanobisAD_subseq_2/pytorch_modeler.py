############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime

# general analysis tool-kit
import numpy as np
import pandas as pd
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
        batch_size=20,
        shuffle=False,
        #num_workers=2,
        #pin_memory=True
        )
    
    valid_source_loader = torch.utils.data.DataLoader(
        dataset=valid_source_dataset,
        batch_size=2,
        shuffle=False,
        #num_workers=2,
        #pin_memory=True
        )
    
    valid_target_loader = torch.utils.data.DataLoader(
        dataset=valid_target_dataset,
        batch_size=2,
        shuffle=False,
        #num_workers=2,
        #pin_memory=True
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




def make_subseq(feature, phase):
    """[summary]

    Args:
        feature (torch.Tensor): (n_mels, time_seq)

    Returns:
        [type]: [description]
    """
    
    n_mels = config['param']['mel_bins']
    n_frames = config['param']['n_frames']
    n_hop_frames = config['param']['n_hop_frames']
    n_vectors = len(feature[0, :]) - n_frames + 1
    dims = n_mels * n_frames
    
    # skip too short clips
    if n_vectors < 1:
        return torch.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = torch.zeros((n_vectors, dims))
    for frame in range(n_frames):
        vectors[:, n_mels * frame : n_mels * (frame + 1)] = feature[
            :, frame : frame + n_vectors
        ].T
    
    #print(vectors.shape)
    
    # reduce sample
    if phase == 'train':
        vectors = vectors[:: n_hop_frames, :]
    
    #print(vectors.shape)
    vectors = vectors.reshape(
        (
            vectors.shape[0],
            1,  # number of channels
            n_frames,
            n_mels,
        )
    )
    return vectors

# training function
def extract_net(net, dataloaders_dict, out_dir, machine_type):
    outputs = []
    def hook(module, input, output):
        #print(output.shape)
        output = output.cpu()
        outputs.append(output.mean(dim=(2,3)))
    
    net.resnet.layer1[-1].register_forward_hook(hook)
    net.resnet.layer2[-1].register_forward_hook(hook)
    net.resnet.layer3[-1].register_forward_hook(hook)
    net.resnet.layer4[-1].register_forward_hook(hook)
    #net.layer1[-1].register_forward_hook(hook)
    #net.layer1[-1].register_forward_hook(hook)
    #net.layer2[-1].register_forward_hook(hook)
    #net.layer3[-1].register_forward_hook(hook)
    #net.layer4[-1].register_forward_hook(hook)
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    #output_dicts = {}
    for phase in ['train', 'valid_source', 'valid_target']:
        net.eval()
        features = []
        labels = []
        wav_names = []
        output_dict = {} # {'features':(n_subseq, time_seq, n_mels), 'labels':labels, 'wav_names':wav_names}
        for sample in tqdm(dataloaders_dict[phase]):
            input = sample['feature']
            batch_size = sample['feature'].shape[0]
            #wav_name = sample['wav_name']
            #label = sample['label'].to('cpu')
            #wav_names = wav_names + wav_name
            
            #plt.imshow(input[0,:,:].cpu().t(), aspect='auto')
            #plt.show()
            #input = input.to(device)
            #if phase == 'train':
            subseq_feats=[]
            for i in range(input.size()[0]):
                # batchごとに取り出す
                per_input = input[i,:,:].t()
                # 部分時系列作成
                per_input = make_subseq(per_input, phase)
                per_input = per_input.to(device)
                subseq_feats.append(per_input)
            # get per_input subseq size
            subseq_size = per_input.size()[0]
            # concat
            subseq_feats = torch.cat(subseq_feats, dim=0).to(device)
            
            with torch.no_grad():
                _ = net(subseq_feats, device)  # (n_subseq, time_seq, n_mels) 
                outputs = torch.cat(outputs, dim=1).cpu().detach().numpy().copy()
                #print(outputs.shape)
                #plt.imshow(outputs, aspect='auto')
                #plt.show()
                for i in range(batch_size):
                    start = i*subseq_size
                    end = (i+1)*subseq_size
                    #print(end-start)
                    output = outputs[start: end, :]
                    label = sample['label'][i].cpu().detach().numpy().copy()
                    wav_name = sample['wav_name'][i]
                    features.append(output)
                    labels.append(label)
                    wav_names.append(wav_name)
                    #per_file_sample = {'features': output,
                    #                   'label': label,
                    #                   'wav_name': wav_name}
                    #output_list.append(per_file_sample)
            
                outputs = []
            #break
        output_dict['features'] = features
        output_dict['label'] = np.stack(labels)
        output_dict['wav_name'] = wav_names
        
        # save
        save_dir = f"{out_dir}/{machine_type}_{phase}_features.pkl"
        pd.to_pickle(output_dict, save_dir)
        logger.info(f'success save features : {save_dir}')
        
            #label = sample['label'].to('cpu')
            #labels.append(label)

            #with torch.no_grad():
            #    _ = net(input)  # (batch_size,input(2D)) 
            #    outputs = torch.cat(outputs, dim=1).cpu()
            #    M_means.append(outputs)
            #    print(outputs.shape)
            #    outputs = []
                #M_means.append(output_dict['M_means'].to('cpu'))
                
        #M_means = torch.cat(M_means, dim=0).detach().numpy().copy()
        #labels = torch.cat(labels, dim=0).detach().numpy().copy()
    
    return 1