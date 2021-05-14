import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
from models import Cnn14_16k
import datasets.mvtec as mvtec
import DCASE2021_task2

# CONFIG
import yaml
import os

with open("./config.yaml", 'rb') as f:
    CONFIG = yaml.load(f)

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def main():
    ##########################################################
    # load model
    # 36行目のところでモデル選択
    # resnet18 or wide_resnet50_2を使う -> 読み込み
    arch = 'PANNs_CNN14_16k'
    model = Cnn14_16k(sample_rate=CONFIG['param']['sample_rate'],
                     window_size=CONFIG['param']['window_size'],
                     hop_size=CONFIG['param']['hop_size'],
                     mel_bins=CONFIG['param']['mel_bins'],
                     fmin=CONFIG['param']['fmin'],
                     fmax=CONFIG['param']['fmax'],
                     classes_num=527,) # なんでもいい
    pretrained_dict = torch.load(CONFIG['IO_OPTION']['PREMODEL_ROOT'])
    model.load_state_dict(pretrained_dict['model'], strict=True)
    t_d = 64    # 全特徴量(64+128+256=448)
    d = 64      # 使う特徴量数
    ##########################################################
    model.to(device)
    model.eval()
    ##########################################################
    ###### seed ##############################################
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)
    ##########################################################
    idx = torch.tensor(sample(range(0, t_d), d)) # t_d(特徴量)の中からランダムにd個サンプリングする(ぜんぶやると重いから) # わからん
    ##########################################################
    # set model's intermediate outputs########################
    # モデルの中間出力の設定
    outputs = []
    # hook functionで中間出力をappendする
    def hook(module, input, output):
        outputs.append(output)

    model.conv_block1.register_forward_hook(hook)
    #model.conv_block2.register_forward_hook(hook)
    #model.conv_block3.register_forward_hook(hook)
    
    # for test
    inputs = []
    def in_hook(module, input, output):
        inputs.append(input)
    
    ###########################################################
    # outputフォルダ作成
    ###########################################################
    os.makedirs(os.path.join(CONFIG['IO_OPTION']['OUTPUT_ROOT'], 'temp_%s' % arch), exist_ok=True)
    ###########################################################
    # 定義
    ###########################################################
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []
    ###########################################################
    # 抽出
    ###########################################################
    # データタイプごとにループ
    for class_name in DCASE2021_task2.CLASS_NAMES:
        # データセット/データローダ作成、
        # is_train = phase に変える
        train_dataset = DCASE2021_task2.DCASE2021_task2_Dataset(CONFIG['IO_OPTION']['INPUT_ROOT'], class_name=class_name, phase='train')
        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['param']['batch_size'], num_workers=2, pin_memory=True)
        test_dataset = DCASE2021_task2.DCASE2021_task2_Dataset(CONFIG['IO_OPTION']['INPUT_ROOT'], class_name=class_name, phase='source_test')
        test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['param']['batch_size'], num_workers=2, pin_memory=True)
        # それぞれのレイヤをdictで管理
        # OrderedDict -> 追加された順番がわかるdict
        #train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        #test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        train_outputs = OrderedDict([('layer1', [])])
        test_outputs = OrderedDict([('layer1', [])])
        # extract train set features (trainの特徴抽出)
        train_feature_filepath = os.path.join(CONFIG['IO_OPTION']['OUTPUT_ROOT'], 'temp_%s' % arch, 'train_%s.pkl' % class_name)
        ############################################# 学習フェーズ #################################################
        # 既に学習がおわっているかどうかをtrain pathをみて判断する(なければ実行)
        if not os.path.exists(train_feature_filepath):
            num_iter = 0
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs(中間層出力の取得)
                # 中間出力をappend
                # key = 'layer1'... value = 値
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    v = v.cpu().detach()
                    if num_iter == 0:
                        train_outputs[k] = v
                    else:
                        train_outputs[k] = torch.cat([train_outputs[k], v], 0)
                num_iter = 1
                # initialize hook outputs
                outputs = []
            # key = 'layer1'... value = 値
            # appendされていたやつをcat
            # train_outputs[k] = (n_sample, Ch, H, W)
            #for k, v in train_outputs.items():
            #    train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            # embedding_concatを使ってパッチごとに特徴を集める
            # layer1を起点にlayer2,layer3の対応する部分を集めてると思われる
            embedding_vectors = train_outputs.pop('layer1')
            #for layer_name in ['layer2', 'layer3']:
            #    embedding_vectors = embedding_concat(embedding_vectors, train_outputs.pop(layer_name))
            #    # del train_outputs
            #    train_outputs[layer_name] = []
            # randomly select d dimension(d次元をランダムに選択)
            # torch.index_select (https://pytorch.org/docs/stable/generated/torch.index_select.html)
            # embedding_vectorsのdim1に沿ってindices = [0,2,4](idx)を取得（サンプルを取得してる)
            # 0~t_d idxの中からランダムにd個サンプリングする
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            ####################### calculate multivariate Gaussian distribution (MVG計算)##########################
            # PaDiMではパッチごとにMVGを作成する
            #print(embedding_vectors.size())
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            # パッチごとにMVG
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution (保存)
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
            ########################################################################################################
        else:
            # すでにtrain pathがあるならロード
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)
        ############################################################################################################
        ############################################# 推論フェーズ #################################################
        # つかわない変数###
        # - mask
        # - gt_mask_list
        # - 
        ###################
        
        # 異常部位のground truthがあれば使う（こんかいはつかわない）
        gt_list = []
        #gt_mask_list = []
        test_imgs = []
        model.logmel_extractor.register_forward_hook(in_hook)
        # extract test set features (trainのときと一緒)##################################################
        for (x, y, wav_name) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            #test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            #gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            #print(inputs[0])
            test_imgs.extend(inputs)
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
            inputs = []

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        #for layer_name in ['layer2', 'layer3']:
        #    embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
        # 時間方向にmeanをしてしまえば、位置情報をきにしなくて良さそう
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        #######################################　マハラノビス距離計算　###############################################
        # calculate distance matrix
        
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        # 対応するMVGを取り出してマハラノビス距離を計算(H*W個のMVGがある、特徴量はC)
        # 一つのMVGを取り出した際には、サンプルすべての対応するパッチMVGとのマハラノビス距離を計算する。
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        
        # 最終的に(B,H,W)の異常スコアマップが出力される
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample(リサイズしたやつをもとにもどす)　# わからん
        dist_list = torch.tensor(dist_list)
        score_map = dist_list.unsqueeze(1).squeeze().numpy()
        # わからん(ノイズ除去？)
        # apply gaussian smoothing on the score map(スコアマップにガウシアン・スムージングを適用する)
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # なんでやるかわからん
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score) # おそらくこれを可視化すればよい（異常部位）
        
        ####################################### 評価 ######################################
        # calculate image-level ROC AUC score
        # 最大値を異常スコアにする
        img_scores = scores.reshape(scores.shape[0], -1).mean(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        ###################################################################################
        
        ######################################### やらない（？）####################################################
        # get optimal threshold(最適な閾値を得る) 
        #gt_mask = np.asarray(gt_mask_list)
        #precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        #a = 2 * precision * recall
        #b = precision + recall
        #f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        #threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC(ピクセル単位のROCAUCを算出)
        #fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        #per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        #total_pixel_roc_auc.append(per_pixel_rocauc)
        #print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        #fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = CONFIG['IO_OPTION']['OUTPUT_ROOT'] + '/' + f'pictures_{arch}'
        os.makedirs(save_dir, exist_ok=True)
        #plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
        plot_fig(test_imgs, scores, save_dir, class_name)

        ############################################################################################################
    
    # 全データセットの平均スコア
    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    # こっちはやらない(異常部位ground truthがないので)################################
    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")
    ##################################################################################

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG['IO_OPTION']['OUTPUT_ROOT'], 'roc_curve.png'), dpi=100)

#def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
def plot_fig(test_img, scores, save_dir, class_name):
    num = len(scores)
    vmax = scores.max()
    vmin = scores.min()
    for i in range(num):
        img = test_img[i]
        #img = denormalization(img)
        #gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i]# * 255
        #mask = scores[i]
        #mask[mask > threshold] = 1
        #mask[mask <= threshold] = 0
        #kernel = morphology.disk(4)
        #mask = morphology.opening(mask, kernel)
        #mask *= 255
        #vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 2, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        # vmin, vmaxをスペクトログラムの場合変えるべきか？
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        # show test_img
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        #ax_img[1].imshow(gt, cmap='gray')
        #ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')
        #ax_img[3].imshow(mask, cmap='gray')
        #ax_img[3].title.set_text('Predicted mask')
        #ax_img[4].imshow(vis_img)
        #ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

# これはMVtech用なので作り直す必要あり
# normalizationしているのでそれをなおしてる
# plot figで使う
# つかわない？
#def denormalization(x):
#    mean = np.array([0.485, 0.456, 0.406])
#    std = np.array([0.229, 0.224, 0.225])
#    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
#    return x

# パッチごとのlayer1,2,3特徴量
# むずい
def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).cuda()
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    
    return z


if __name__ == '__main__':
    main()
