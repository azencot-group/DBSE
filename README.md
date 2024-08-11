# DBSE
Sequential **D**isentanglement **b**y Extracting Static Information From A **S**ingle Sequence **E**lement

### [Paper](https://arxiv.org/pdf/2406.18131)

## ℹ️ Overview
This project presents a novel approach to unsupervised sequential disentanglement by using a simple architecture that mitigates information leakage through a subtraction inductive bias, conditioning on a single sample instead of the entire sequence. Our method achieves state-of-the-art results on multiple data-modality benchmarks, simplifying the variational framework with fewer loss terms, hyperparameters, and data augmentation requirements.

<img width="1105" alt="image" src="https://github.com/user-attachments/assets/e4133951-1767-4de2-bea2-7ddeb78720d7">

## Setup
Download and set up the repository:
```bash
git clone https://github.com/azencot-group/DBSE.git
```

## 📊 Data
[Sprites](https://paperswithcode.com/dataset/sprites)<br>
[Time Series](https://drive.google.com/drive/folders/1bzECwhWXtCrgwOHBzcIlCMVYLr6OGi56?usp=sharing)<br>
[Timit](https://catalog.ldc.upenn.edu/LDC93S1)<br><br>
<b>Note:</b> The [Mug](https://www.researchgate.net/publication/224187946_The_MUG_facial_expression_database) dataset is private, so we are unable to upload it here.

## Video

For Training and Evaluation of mug dataset (Table 1 in our paper):
```bash
cd video/mug/
python run_mug.py
```

For Training and Evaluation of sprites dataset (Table 1 in our paper):
```bash
cd video/sprites/
python run_sprites.py
```

## Time Series

For Training and Evaluation of etth1 predictor (Table 3 in our paper):
```bash
cd time_series/etth1_prediction/
python train_e2e_etth1_prediction.py
```

For Training and Evaluation of physionet predictor (Table 3 in our paper):
```bash
cd time_series/physionet_prediction/
python train_e2e_physionet_prediction.py
```

For Training and Evaluation of air quality classifier (Table 4 in our paper):
```bash
cd time_series/air_quality_classifier/
python train_e2e_air_quality_classifier.py
```

For Training and Evaluation of physionet classifier (Table 4 in our paper):
```bash
cd time_series/physionet_classifier/
python train_e2e_physionet_classifier.py
```

For Training and Evaluation of timit classifier (Table 7 in our paper):
```bash
cd audio/
python train_e2e_timit.py
```

## Bibtex:
Please cite our paper, if you happen to use this codebase:

```
@inproceedings{berman2024DBSE,
  title={Sequential Disentanglement by Extracting Static Information From A Single Sequence Element},
  author={Nimrod Berman, Ilan Naiman, Idan Arbiv, Gal Fadlon, Omri Azencot},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
