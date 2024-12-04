# DBSE
Sequential **D**isentanglement **b**y Extracting Static Information From A **S**ingle Sequence **E**lement (ICML 2024)

### [Paper](https://arxiv.org/pdf/2406.18131)

## ℹ️ Overview
This project presents a novel approach to unsupervised sequential disentanglement by using a simple architecture that mitigates information leakage through a subtraction inductive bias, conditioning on a single sample instead of the entire sequence. Our method achieves state-of-the-art results on multiple data-modality benchmarks, simplifying the variational framework with fewer loss terms, hyperparameters, and data augmentation requirements.

<img width="1105" alt="image" src="https://github.com/user-attachments/assets/e4133951-1767-4de2-bea2-7ddeb78720d7">

## Setup
Download and set up the repository:
```bash
git clone https://github.com/azencot-group/DBSE.git
```

### Conda Environment Setup
To create the Conda environment from the `environment.yml` file, run:
```bash
conda env create -f environment.yml
```

### Pip Environment Setup
If you're using Pip, create a `requirements.txt` file manually from the dependencies, then install them:
```bash
pip install -r requirements.txt
```

## 📊 Data
[Downloag Data](https://drive.google.com/drive/folders/1bzECwhWXtCrgwOHBzcIlCMVYLr6OGi56?usp=sharing)<br><br>
<b>Note:</b> The [Mug](https://www.researchgate.net/publication/224187946_The_MUG_facial_expression_database) dataset is private, so we are unable to upload it here.

## Video

For Training and Evaluation of mug dataset (Table 1 in our paper):
- Implement define_classifier and load_dataset in mug_utils.py.
```bash
cd video/mug/
python run_mug.py
```

For Training and Evaluation of sprites dataset (Table 1 in our paper):
- Ensure the --dataset_path argument is set in sprites_hyperparameters.py.

```bash
cd video/sprites/
python run_sprites.py
```

## Time Series

For Training and Evaluation of etth1 predictor (Table 3 in our paper):
- Add the --dataset_path argument in train_and_eval_etth1_prediction.py.
```bash
cd time_series/etth1_prediction/
python train_e2e_etth1_prediction.py
```

For Training and Evaluation of physionet predictor (Table 3 in our paper):
- Add --data_dir, --physionet_dataset_path, and --physionet_static_dataset_path in train_and_eval_physionet_prediction.py.
```bash
cd time_series/physionet_prediction/
python train_and_eval_physionet_prediction.py
```

For Training and Evaluation of air quality classifier (Table 4 in our paper):
- Add the --dataset_path argument in train_and_eval_air_quality_classifier.py.
```bash
cd time_series/air_quality_classifier/
python train_and_eval_air_quality_classifier.py
```

For Training and Evaluation of physionet classifier (Table 4 in our paper):
- Add --data_dir, --physionet_dataset_path, and --physionet_static_dataset_path in train_and_eval_physionet_classifier.py.
```bash
cd time_series/physionet_classifier/
python train_and_eval_physionet_classifier.py
```
## Audio

For Training and Evaluation of timit classifier (Table 7 in our paper):
- Ensure the --dataset_path argument is set in timit_hyperparameters.py.
```bash
cd audio/
python train_and_eval_timit.py
```

<b>Note:</b> The files already contain the hyperparameters we used when reporting the results in the paper.


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
