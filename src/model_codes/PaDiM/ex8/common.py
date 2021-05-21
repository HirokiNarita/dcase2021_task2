########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import random
import time
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG

# additional
import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
import librosa
import librosa.core
import librosa.feature
import yaml
import torch
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score

########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################

########################################################################
# load parameter.yaml
########################################################################
def yaml_load(path="./param.yaml"):
    with open(path) as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################
# logger
########################################################################
def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_folder) #fh = file handler
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger

########################################################################
# file I/O
########################################################################
# wav file input
def file_load(wav_name, sr=16000, mono=True):
    """
    load .wav file.

    wav_name : str
        target .wav file
   sr : int
        sampling rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=sr, mono=mono)
    except:
        print("file_broken or not exists!! : {}".format(wav_name))

########################################################################
# mel spectrogram generator
########################################################################
def log_melspec_generate(file_name,
                         n_mels=64,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    
    # generate melspectrogram using librosa
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))
    
    return log_mel_spectrogram

########################################################################

import matplotlib.pyplot as plt
########################################################################
# feature extractor
########################################################################
def file_to_vectors_2d(file_name,
                       n_mels=128,
                       n_frames=64,
                       n_fft=1024,
                       hop_length=512,
                       power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    #dims = n_mels * n_frames
    #n_frames=64
    # generate melspectrogram using librosa
    waveform, sample_rate = file_load(file_name, mono=True)
    mel_spectrogram_transformer = T.MelSpectrogram(sample_rate=sample_rate,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   n_mels=n_mels,
                                                   power=power)
    mel_spectrogram = mel_spectrogram_transformer(waveform)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * torch.log10(torch.maximum(mel_spectrogram))
    #plt.imshow(log_mel_spectrogram, aspect='auto')
    #plt.show()
    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return torch.empty((0, n_frames, n_mels))
    
    # generate feature vectors by concatenating multiframes
    vectors = torch.zeros((n_vectors, n_frames, n_mels))
    for t in range(n_vectors):
        #vectors[:, n_frames * t : n_frames * (t + 1), n_mels] = log_mel_spectrogram[:, t : t + n_vectors].T
        vectors[t, :, :] = log_mel_spectrogram[:, t : t+n_frames].T

    return vectors


########################################################################


########################################################################
# get directory paths according to mode
########################################################################
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


########################################################################


########################################################################
# get machine IDs
########################################################################
def get_section_names(target_dir,
                      dir_name,
                      ext="wav"):
    """
    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files

    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return section_names


########################################################################


########################################################################
# get the list of wave file paths
########################################################################
def file_list_generator(target_dir,
                        section_name,
                        dir_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))

    # development
    if mode:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_normal,
                                                                                                     ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_anomaly,
                                                                                                     ext=ext))
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     ext=ext))
        files = sorted(glob.glob(query))
        labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels


########################################################################


########################################################################
# tic toc functions
########################################################################
def tic():
    #require to import time
    global start_time_tictoc
    start_time_tictoc = time.time()

def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.9f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")

def get_section_types(wav_names):
    """
    wav_nameリストから
    セクションタイプリストを得る関数

    Args:
        wav_names (list): 音源ファイルのパスリスト

    Returns:
        np.array: sectionタイプのone-hot
    """
    section_types = []
    for wav_name in wav_names:
        if 'section_00' in wav_name:
            section_types.append(0)
        elif 'section_01' in wav_name:
            section_types.append(1)
        elif 'section_02' in wav_name:
            section_types.append(2)
        elif 'section_03' in wav_name:
            section_types.append(3)
        elif 'section_04' in wav_name:
            section_types.append(4)
        else:
            section_types.append(5)
    
    return np.array(section_types)

def get_target_binary(wav_names):
    """
    wav_nameリストからtargetか
    否かのone-hotベクトルを得る関数

    Args:
        wav_names (list): 音源ファイルのパスリスト

    Returns:
        np.array: 0 or 1のone-hotベクトル
    """
    targets_binary = []
    for wav_name in wav_names:
        if 'target' in wav_name:
            targets_binary.append(1)
        else:
            targets_binary.append(0)
    
    return np.array(targets_binary)

def get_label(file_path):
    if "normal" in file_path:
        label = 0
    else:
        label = 1
    return label

def calc_auc(y_true, y_pred, max_fpr=0.1):
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    return auc, p_auc

def get_pred_discribe(labels, preds, section_types):
    #print(labels.shape)
    #print(preds.shape)
    #print(section_types.shape)
    describe_df = pd.DataFrame(np.stack([labels, preds, section_types], axis=1),
                                columns=['labels', 'preds', 'section_types'])
    describe_df = describe_df.astype({'labels': int, 'section_types': int})
    return describe_df

def get_score_per_Section(describe_df, max_fpr=0.1):
    # ユニークsectionを取得、昇順ソート
    sections = np.sort(describe_df['section_types'].unique())

    for section in sections:
        per_section_df = describe_df[describe_df['section_types'] == section]
        per_section_AUC = roc_auc_score(per_section_df['labels'], per_section_df['preds'])
        per_section_pAUC = roc_auc_score(per_section_df['labels'], per_section_df['preds'], max_fpr=max_fpr)
        # column = [AUC,pAUC], row = index
        score_df = pd.DataFrame(np.stack([per_section_AUC, per_section_pAUC]), index=['AUC', 'pAUC']).T
        # indexをsectionナンバーにrename
        # column = [AUC,pAUC], row = [section]
        score_df.index = [section]
        if section == 0:
            scores_df = score_df.copy()
        else:
            # 結合
            scores_df = scores_df.append(score_df)
    return scores_df

def calc_DCASE2021_score(all_scores_df, labels, preds, section_types, phase):
    describe_df = get_pred_discribe(labels, preds, section_types)
    scores_df = get_score_per_Section(describe_df, max_fpr=0.1)
    # 結合(source + target)
    if phase == 'valid_source':
        scores_df = scores_df.rename(index=lambda num: 'Source_' + f'{num}')
        all_scores_df = scores_df.copy()
    elif phase == 'valid_target':
        scores_df = scores_df.rename(index=lambda num: 'Target_' + f'{num}')
        all_scores_df = all_scores_df.append(scores_df)
        # 平均
        mean_df = pd.DataFrame(all_scores_df.mean(axis=0)).T
        mean_df.index = ['mean']
        # 調和平均
        hmean = scipy.stats.hmean(all_scores_df, axis=0)
        hmean_df = pd.DataFrame(hmean, index=['AUC', 'pAUC']).T
        hmean_df.index = ['h_mean']
        # 結合
        all_scores_df = all_scores_df.append(mean_df)
        all_scores_df = all_scores_df.append(hmean_df)
        # 出力
        #all_scores_df.to_csv(f'{OUT_SCORE_DIR}/{machine_type}_score.csv')
        
    return all_scores_df
