# DCASE2021 task2

# Task description
http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds

# Schedule
- Task open: 1st of March 2021
- Additional training dataset release: __1st of April 2021__
- Evaluation dataset release: __1st of June 2021__
- External resource list lock: __1st of June 2021__
- Challenge deadline: __15th of June 2021__
- Challenge results: 1st of July 2021

# environment
- Anaconda=4.9.2
- librosa=0.8.0
- torch=1.8
- torchlibrosa=0.0.8

# This repository
- EDA
    - /src/EDA/view_melspec.ipynb
        - normalのsourceとtargetのスペクトログラムを可視化
- MahalanobisAD
    - /src/model_codes/MahalanobisAD
    - https://arxiv.org/abs/2005.14140
    - pre-trainedモデルにはPANNs ResNet38を用いた
        - https://arxiv.org/abs/1912.10211

