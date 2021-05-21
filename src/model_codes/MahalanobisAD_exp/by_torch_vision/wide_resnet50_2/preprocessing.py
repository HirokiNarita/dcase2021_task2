########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
import librosa
import torchaudio.transforms as T
import torch
import cv2
#from torchaudio.transforms import Resample
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

class extract_waveform(object):
    """
    データロード(波形)
    
    Attributes
    ----------
    sound_data : logmelspectrogram
    """
    def __init__(self, sound_data=None):
        self.sound_data = sound_data
        self.img_size = 224
    
    def __call__(self, sample):

        sample_rate=config['param']['sample_rate']
        n_mels = config['param']['mel_bins']
        n_fft = config['param']['window_size']
        hop_length=config['param']['hop_size']
        power = 2.0
        
        #self.resampling = Resample(16000, sample_rate)
        #input = self.resampling(input)
        audio, sample_rate = librosa.load(sample['wav_name'],
                                          sr=config['param']['sample_rate'],
                                          mono=True)
        audio = torch.from_numpy(audio.astype(np.float32)).clone().cuda()
        mel_spectrogram_transformer = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=None,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        ).cuda()
        X = mel_spectrogram_transformer(audio).cpu()
        eps = 1e-16
        X = (
            20.0 / power * torch.log10(X + eps)
        )
        ##### for imagenet model####
        X = X.detach().numpy().copy()
        X = self.mono_to_color(X)
        height, width, _ = X.shape
        X = cv2.resize(X, (int(width * self.img_size / height), self.img_size))
        X = np.moveaxis(X, 2, 0)
        X = (X / 255.0).astype(np.float32)
        X = torch.from_numpy(X.astype(np.float32)).clone()
        ############################
        self.sound_data = X
        self.label = np.array(sample['label'])
        self.wav_name = sample['wav_name']
        
        return {'feature': self.sound_data, 'label': self.label, 'wav_name': self.wav_name}

    def mono_to_color(self,
                      X: np.ndarray,
                      mean=None,
                      std=None,
                      norm_max=None,
                      norm_min=None,
                      eps=1e-6):
        """
        Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
        """
        # Stack X as [X,X,X]
        X = np.stack([X, X, X], axis=-1)

        # Standardize
        mean = mean or X.mean()
        X = X - mean
        std = std or X.std()
        Xstd = X / (std + eps)
        _min, _max = Xstd.min(), Xstd.max()
        norm_max = norm_max or _max
        norm_min = norm_min or _min
        if (_max - _min) > eps:
            # Normalize to [0, 255]
            V = Xstd
            V[V < norm_min] = norm_min
            V[V > norm_max] = norm_max
            V = 255 * (V - norm_min) / (norm_max - norm_min)
            V = V.astype(np.uint8)
        else:
            # Just zero
            V = np.zeros_like(Xstd, dtype=np.uint8)
        return V
    
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        feature, label, wav_name = sample['feature'], sample['label'], sample['wav_name']
        return {'feature': feature, 'label': torch.from_numpy(label), 'wav_name': wav_name}



class DCASE_task2_Dataset(torch.utils.data.Dataset):
    '''
    Attribute
    ----------
    
    '''
    
    def __init__(self, file_list, transform=None):
        self.transform = transform
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # ファイル名でlabelを判断
        if "normal" in file_path:
            label = 0
        else:
            label = 1
        
        sample = {'wav_name':file_path, 'label':np.array(label)}
        sample = self.transform(sample)
        
        return sample
