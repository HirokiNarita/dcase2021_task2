########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

class extract_waveform(object):
    """
    wavデータロード(波形)
    
    Attributes
    ----------
    sound_data : waveform
    """
    def __init__(self, sound_data=None):
        self.sound_data = sound_data
    
    def __call__(self, sample):
        self.sound_data = com.file_load(sample['wav_name'],
                                        sr=config['param']['sample_rate'],
                                        mono=True)
        self.sound_data = self.sound_data[0]
        
        self.label = sample['label']
        self.section_type = sample['type']
        self.wav_name = sample['wav_name']
        
        return {'feature': self.sound_data,
                'label': self.label,
                'type': self.section_type,
                'wav_name': self.wav_name}

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        feature, label, section_type, wav_name = sample['feature'], sample['label'], sample['type'], sample['wav_name']
        
        return {'feature': torch.from_numpy(feature).float(),
                'label': torch.from_numpy(label).long(),
                'type': torch.from_numpy(section_type).long(),
                'wav_name': wav_name}

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
        section_type = self.get_section_type(file_path)
        label = self.get_label(file_path)
        
        sample = {'wav_name':file_path, 'label':label, 'type':section_type}
        sample = self.transform(sample)
        
        return sample

    def get_section_type(self, file_path):
        if 'section_00' in file_path:
            section_type = 0
        elif 'section_01' in file_path:
            section_type = 1
        elif 'section_02' in file_path:
            section_type = 2
        elif 'section_03' in file_path:
            section_type = 3
        elif 'section_04' in file_path:
            section_type = 4
        elif 'section_05' in file_path:
            section_type = 5
    
        return np.array(section_type)
    
    def get_label(self, file_path):
        if "normal" in file_path:
            label = 0
        else:
            label = 1
            
        return np.array(label)