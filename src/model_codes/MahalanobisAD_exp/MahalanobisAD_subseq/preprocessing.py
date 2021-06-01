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
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchaudio.transforms import Resample
import torchaudio
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
        
        self.sample_rate=config['param']['sample_rate']
        self.mel_bins = config['param']['mel_bins']
        self.window_size = config['param']['window_size']
        self.hop_size=config['param']['hop_size']
        self.fmin=config['param']['fmin']
        self.fmax=config['param']['fmax']
        self.power = 2.0
        self.window = 'hann'
        self.center = True
        self.pad_mode = 'reflect'
        self.ref = 1.0
        self.amin = 1e-10
        self.top_db = None
        
        self.resampling = Resample(16000, self.sample_rate).cuda()
        
        self.spectrogram_extractor = Spectrogram(n_fft=self.window_size, hop_length=self.hop_size, 
            win_length=self.window_size, window=self.window, center=self.center, pad_mode=self.pad_mode, 
            freeze_parameters=True).cuda()

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=self.sample_rate, n_fft=self.window_size, 
            n_mels=self.mel_bins, fmin=self.fmin, fmax=self.fmax, ref=self.ref, amin=self.amin, top_db=self.top_db, 
            freeze_parameters=True).cuda()
    
    def __call__(self, sample):
        x = torchaudio.load(sample['wav_name'])[0].cuda()
        x = self.resampling(x)
        #self.resampling = Resample(16000, sample_rate)
        #input = self.resampling(input)
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)    #(batch_size, 1, time_steps, freq_bins)
        x = x.squeeze().cpu()           #(batch_size, time_steps, freq_bins)

        self.sound_data = x
        self.label = np.array(sample['label'])
        self.wav_name = sample['wav_name']
        
        return {'feature': self.sound_data, 'label': self.label, 'wav_name': self.wav_name}
    
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
