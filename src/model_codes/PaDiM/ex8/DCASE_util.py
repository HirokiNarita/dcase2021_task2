# Standard library imports.
import os
import sys
from scipy.stats.stats import describe
import yaml

import joblib

# Related third party imports.
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.utils.data
from scipy.special import softmax
from sklearn import metrics

# Local application/library specific imports.
import util
import DCASE2021_task2

with open("./config.yaml", 'rb') as f:
    CONFIG = yaml.load(f)

def calc_evaluation_scores(y_true, y_pred, decision_threshold):
    """
    Calculate evaluation scores (AUC, pAUC, precision, recall, and F1 score)
    """
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=CONFIG['param']["max_fpr"])

    (_, false_positive, false_negative, true_positive,) = metrics.confusion_matrix(
        y_true, [1 if x > decision_threshold else 0 for x in y_pred]
    ).ravel()

    prec = true_positive / np.maximum(
        true_positive + false_positive, sys.float_info.epsilon
    )
    recall = true_positive / np.maximum(
        true_positive + false_negative, sys.float_info.epsilon
    )
    f1_score = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)

    print("AUC : {:.6f}".format(auc))
    print("pAUC : {:.6f}".format(p_auc))
    print("precision : {:.6f}".format(prec))
    print("recall : {:.6f}".format(recall))
    print("F1 score : {:.6f}".format(f1_score))

    return auc, p_auc, prec, recall, f1_score

def calc_performance_section(performance, csv_lines):
    """
    Calculate model performance per section.
    """
    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
    hmean_performance = scipy.stats.hmean(
        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
        axis=0,
    )
    csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
    csv_lines.append([])

    return csv_lines

def calc_performance_all(performance, csv_lines):
    """
    Calculate model performance over all sections.
    """
    csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(
        ["arithmetic mean over all machine types, sections, and domains", ""]
        + list(amean_performance)
    )
    hmean_performance = scipy.stats.hmean(
        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
        axis=0,
    )
    csv_lines.append(
        ["harmonic mean over all machine types, sections, and domains", ""]
        + list(hmean_performance)
    )
    csv_lines.append([])

    return csv_lines


class DCASE2021_Task2_Score_Calculator:
    """
    ・異常スコアの計算、提出形式の異常スコアcsv作成
    ・mode == 'dev': scoreの計算
    """
    
    def __init__(self, class_name, mode='dev'):
        self.mode = mode
        self.class_name = class_name
        if self.mode == 'dev':
            self.describe_df = pd.DataFrame(columns=['wav_name', 'pred', 'label'])
            self.describe_df = self.describe_df.astype({'wav_name': str, 'pred': float, 'label': int})
        else:
            self.describe_df = pd.DataFrame(columns=['wav_name', 'pred'])
            self.describe_df = self.describe_df.astype({'wav_name': str, 'pred': float})
    
    def append(self, wav_name, pred, label=None):
        if self.mode == 'dev':
            per_describe = pd.DataFrame(
                data=np.stack([wav_name, pred, label], axis=1),
                columns=self.describe_df.columns,
                )
        else:
            per_describe = pd.DataFrame(
                data=np.stack([wav_name, pred], axis=1),
                columns=self.describe_df.columns,
                )

        self.describe_df = self.describe_df.append(per_describe, ignore_index=True)

    def calc_score(self, out_file_name):
        wav_name_list = self.describe_df['wav_name'].to_list()
        section_types = self.get_section_types(wav_name_list)
        tgt_bool = self.get_target_binary(wav_name_list)

        tmp_describe_df = self.describe_df
        tmp_describe_df['section'] = section_types
        tmp_describe_df['tgt_bool'] = tgt_bool
        score_df = pd.DataFrame(columns=['section', 'tgt_bool', 'AUC', 'pAUC'])
        #score_df.astype({'section':int, 'tgt_bool':int})
        for section in tmp_describe_df['section'].unique():
            for tgt_bool in tmp_describe_df['tgt_bool'].unique():
                per_sec_df = tmp_describe_df[(tmp_describe_df['section'] == section) & (tmp_describe_df['tgt_bool'] == tgt_bool)]
                AUC = metrics.roc_auc_score(per_sec_df['label'],
                                            per_sec_df['pred'],
                                            )
                pAUC = metrics.roc_auc_score(per_sec_df['label'],
                                            per_sec_df['pred'],
                                            max_fpr=CONFIG['param']['max_fpr'],
                                            )
                tmp_score_df = pd.Series([section, tgt_bool, AUC, pAUC],
                                         index=score_df.columns)
                score_df = score_df.append(tmp_score_df, ignore_index=True)
                print(f'Section {section} tgt{tgt_bool} : AUC {AUC:.3f}, pAUC {pAUC:.3f}')
        
        mean_AUC = score_df['AUC'].values.mean()
        mean_pAUC = score_df['pAUC'].values.mean()
        hmean_AUC = scipy.stats.hmean(score_df['AUC'].values)
        hmean_pAUC = scipy.stats.hmean(score_df['pAUC'].values)
        tmp_score_df = pd.Series(['mean', 'None', mean_AUC, mean_pAUC],
                                 index=score_df.columns)
        score_df = score_df.append(tmp_score_df, ignore_index=True)
        tmp_score_df = pd.Series(['h_mean', 'None', hmean_AUC, hmean_pAUC],
                                 index=score_df.columns)
        score_df = score_df.append(tmp_score_df, ignore_index=True)
        # save .csv
        score_df.to_csv(out_file_name, index=False)
    
    def save_anomaly_score(self, file_name):
        self.describe_df = self.describe_df.drop('label')
        self.describe_df.to_csv(file_name, index=False)
    
    def get_target_binary(self, wav_names):
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
    
    def get_section_types(self, wav_names):
        """
        wav_nameリストから
        セクションタイプリストを得る関数

        Args:
            wav_names (list): 音源ファイルのパスリスト

        Returns:
            np.array: sectionタイプ
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

