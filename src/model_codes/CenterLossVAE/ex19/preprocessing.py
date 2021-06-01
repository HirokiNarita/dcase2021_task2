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

class DCASE_task2_Dataset(torch.utils.data.Dataset):
    '''
    Attribute
    ----------
    
    '''
    
    def __init__(self, ext_data):
        self.features = ext_data['features']
        self.wav_name = ext_data['wav_names']
        self.label = ext_data['labels']
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        wav_name = self.wav_name[idx]
        # ファイル名でlabelを判断
        features = torch.from_numpy(self.features[idx]).float()
        section_type = torch.from_numpy(self.get_section_type(wav_name)).long()
        target_bool = torch.from_numpy(self.get_target_bool(wav_name)).long()
        label = self.label[idx]
    
        sample = {'features':features, 'label':label, 'type':section_type, 'target_bool':target_bool, 'wav_name':wav_name}
        
        return sample
    
    def get_section_type(self, wav_name):

        if 'section_00' in wav_name:
            section_type = 0
        elif 'section_01' in wav_name:
            section_type = 1
        elif 'section_02' in wav_name:
            section_type = 2
        elif 'section_03' in wav_name:
            section_type = 3
        elif 'section_04' in wav_name:
            section_type = 4
        elif 'section_05' in wav_name:
            section_type = 5

        return np.array(section_type)
    
    def get_target_bool(self, wav_name):
        if 'target' in wav_name:
            target_bool = 1
        else:
            target_bool = 0
        return np.array(target_bool)