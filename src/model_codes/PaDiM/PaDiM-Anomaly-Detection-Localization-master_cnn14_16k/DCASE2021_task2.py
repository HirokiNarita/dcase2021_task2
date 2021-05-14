import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset
# torchaudio
import torchaudio
from torchaudio.transforms import Resample


from torchvision import transforms as T

# CONFIG
import yaml
import os

with open("./config.yaml", 'rb') as f:
    CONFIG = yaml.load(f)

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
# main.py :96　でよばれる
#CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carCLASS_NAMESpet', 'grid',
#               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

CLASS_NAMES = ['ToyCar', 'ToyTrain', 'fan',
               'gearbox', 'pump', 'slider', 'valve']

# Datasetでclass_nameを定義してる（べんりそう)
class DCASE2021_task2_Dataset(Dataset):
    def __init__(self, dataset_path=CONFIG['IO_OPTION']['INPUT_ROOT'], class_name='ToyCar', phase='train'):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.phase = phase  # 'train' or 'source_test', or 'target_test'
        #self.resize = resize
        #self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.wav_names = self.load_dataset_folder()
        if phase == 'train':
            self.wav_names = self.wav_names[:1000]
        else:
            self.wav_names = self.wav_names[:200]
        # set transforms
        #self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
        #                              T.CenterCrop(cropsize),
        #                              T.ToTensor(),
        #                              T.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])])
        #self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
        #                                 T.CenterCrop(cropsize),
        #                                 T.ToTensor()])
        self.resampling = Resample(16000, CONFIG['param']['sample_rate'])
        
    def __getitem__(self, idx):
        wav_name = self.wav_names[idx]
        x = torchaudio.load(wav_name)[0]
        x = self.resampling(x)[0]
        y = self.get_label(wav_name)
        
        return x, y, wav_name

    def __len__(self):
        return len(self.wav_names)

    def load_dataset_folder(self):
        
        wav_dir = os.path.join(self.dataset_path, self.class_name, self.phase)
        wav_types = sorted(os.listdir(wav_dir))
        wav_types = [f"{wav_dir}/{file}" for file in wav_types]
        return list(wav_types)
    
    def get_label(self, wav_name):
        if 'normal' in wav_name:
            y = 0
        else:
            y = 1
            
        return torch.from_numpy(np.array(y))
#     def download(self):
#         """Download dataset if not exist"""

#         if not os.path.exists(self.mvtec_folder_path):
#             tar_file_path = self.mvtec_folder_path + '.tar.xz'
#             if not os.path.exists(tar_file_path):
#                 download_url(URL, tar_file_path)
#             print('unzip downloaded dataset: %s' % tar_file_path)
#             tar = tarfile.open(tar_file_path, 'r:xz')
#             tar.extractall(self.mvtec_folder_path)
#             tar.close()

#         return


# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, output_path):
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
