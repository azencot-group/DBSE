import sys

sys.path.append('../models')
import random
from speechbrain.dataio.dataio import write_audio
import utils
import progressbar
import numpy as np
import os
import argparse
import itertools
import scipy
from torchaudio import transforms as ts
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from torch.utils.data import DataLoader
import speechbrain as sb
from model.dbse import DBSE
import matplotlib.pyplot as plt
import neptune.new as neptune
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--nEpoch', default=500, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')
parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')
parser.add_argument('--neptune', default=True, type=bool, help='activate neptune tracking')

parser.add_argument('--dataset', default='TIMIT', type=str, help='dataset to train')
parser.add_argument("--dataset_path", default='Your data set path')
parser.add_argument('--decoder', default='ConvT', type=str, help='Upsampling+Conv or Transpose Conv: Conv or ConvT')

parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
parser.add_argument('--f_dim', default=128, type=int, help='dim of f')
parser.add_argument('--z_dim', default=128, type=int, help='dimensionality of z_t')
parser.add_argument('--g_dim', default=128, type=int, help='dimensionality of encoder output vector and decoder input vector')

# audio args
# parser.add_argument('--w_len', type=int, default=160, help='window length')
# parser.add_argument('--h_len', type=int, default=160, help='hop length')
parser.add_argument('--w_len', type=int, default=320, help='window length')
parser.add_argument('--h_len', type=int, default=165, help='hop length')
parser.add_argument('--power', type=int, default=1, help='power of the spectrogram')
parser.add_argument('--fft_size', type=int, default=400, help='power of the spectrogram')

parser.add_argument('--loss_recon', default='L2', type=str, help='reconstruction loss: L1, L2')
parser.add_argument('--note', default='mug', type=str, help='appx note')
parser.add_argument('--weight_f', default=1, type=float, help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', default=1, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_rec', default=20, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_rec_frame', default=100, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--sche', default='const', type=str, help='scheduler')
parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
parser.add_argument('--tag', type=str, default='koopman')
parser.add_argument('--niter', type=int, default=5, help='number of runs for testing')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

mse_loss = nn.MSELoss().cuda()

def permute(data):
    c_aug = torch.zeros(data.shape)
    for idx, sample in enumerate(data):
        perm = torch.randperm(sample.shape[0])
        c_aug[idx] = sample[perm]
    return c_aug


def audio_path_to_data(X):
    """The pipline:
        1. we import the audio wav file in size (t x 1)
        2. we chunk the audio to 200ms frames = size(3200 x 1)
        3. we pad all frames shorter then 200ms with zeros
        4. we chunk it into a new batch = (new_batch_size x 3200)
        5. """
    # load audio
    audio_list = [sb.dataio.dataio.read_audio(x) for x in X]
    max_len = max([len(x) for x in audio_list])

    # pad signals
    padded_list = [torch.concat((x, torch.zeros(max_len - x.shape[0]))) for x in audio_list]

    return torch.stack(padded_list, dim=0), [len(x) // 165 for x in audio_list]


def audio_path_to_data_dbse(X):
    """The pipline:
        1. we import the audio wav file in size (t x 1)
        2. we chunk the audio to 200ms frames = size(3200 x 1)
        3. we pad all frames shorter then 200ms with zeros
        4. we chunk it into a new batch = (new_batch_size x 3200)
        5. """
    # load audio
    # audio_list = [sb.dataio.dataio.read_audio(x) for x in X]
    audio_list = []
    chunck_nums = []
    for x in X:
        full_audio = sb.dataio.dataio.read_audio(x)
        audio_chunks = list(torch.split(full_audio, 3200))
        audio_chunks[-1] = torch.concat((audio_chunks[-1], torch.zeros(3200 - audio_chunks[-1].shape[0])))
        audio_list = audio_list + audio_chunks
        chunck_nums.append(len(audio_chunks))

    return torch.stack(audio_list, dim=0), chunck_nums


def audio_path_to_data_no_mi(X):
    """The pipline:
        1. we import the audio wav file in size (t x 1)
        2. we chunk the audio to 200ms frames = size(3200 x 1)
        3. we pad all frames shorter then 200ms with zeros
        4. we chunk it into a new batch = (new_batch_size x 3200)
        5. """
    # load audio
    # audio_list = [sb.dataio.dataio.read_audio(x) for x in X]
    audio_list = []
    max_len = -1
    for x in X:
        full_audio = sb.dataio.dataio.read_audio(x)
        audio_list.append(full_audio)
        max_len = max(max_len, full_audio.shape[0])

    padded_audio_list = []
    for x in audio_list:
        padded_audio_list.append(torch.concat((x, torch.zeros(max_len - x.shape[0]))))

    return torch.stack(padded_audio_list, dim=0)


def voice_verification_dbse(dbse, spectrogram, test_loader, run):
    for epoch in range(opt.niter):

        print("Epoch", epoch)
        dbse.eval()
        # mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        # mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        # pred1_all, pred2_all, label2_all = list(), list(), list()
        # label_gt = list()

        dataset = list(test_loader.dataset)
        wavs, chunck_nums = audio_path_to_data_dbse([w['wav'] for w in dataset])

        X = wavs
        # encode
        X = X.cuda()
        X = spectrogram(X)
        X = X.permute(0, 2, 1)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
        recon_x, _ = dbse(X)

        # plt.imshow(X[0:28].reshape(-1, 201).cpu())
        # plt.savefig('x1_spec')

        # plt.imshow(recon_x[0:28].reshape(-1, 201).cpu())
        # plt.savefig('x1_rec_spec')

        # plt.plot(spectrogram(wavs[0:28].reshape(-1).cuda()).cpu())
        # plt.show()

        # plt.imshow(X[67:67+15].reshape(-1, 201).cpu())
        # plt.savefig('x_spec')

        n = 0
        f_means_all = []
        for num in chunck_nums:
            f_means_all.append(torch.mean(f_post[n:n + num], dim=0))
            n += num
        f_post_mean = torch.stack(f_means_all)

        n = 0
        z_means_all = []
        for num in chunck_nums:
            z_means_all.append(torch.mean(torch.mean(z_post[n:n + num], dim=0), dim=0))
            n += num
        z_post_mean = torch.stack(z_means_all)

        # f_post_mean = f_post
        # z_post_mean = torch.stack(z_post_mean)  # DEBUG AND SEE IF IT ACTUALLY MEAN VALUE OF THE SEQUENCE DIM

        # --- create pairs of verifications and expected output ---
        index_comb = list(itertools.combinations(range(len(dataset)), 2))

        static_pairs = []
        dynamic_pairs = []
        dataset_list = list(dataset)
        # take the frames for each pair of samples
        for comb in index_comb:
            static_pairs.append(
                [f_post_mean[comb[0]], f_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])
            dynamic_pairs.append(
                [z_post_mean[comb[0]], z_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])

        # --- def for binary search ---
        def f_s(epsilon):
            f_p_static = 0
            f_n_static = 0
            positive = 0
            negative = 0
            for pair in static_pairs:
                decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_static = f_p_static + 1
                elif not decision and pair[2]:
                    f_n_static = f_n_static + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_static / positive) - (f_p_static / negative)

        def f_d(epsilon):
            f_p_dyn = 0
            f_n_dyn = 0
            positive = 0
            negative = 0
            for pair in dynamic_pairs:
                decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_dyn = f_p_dyn + 1
                elif not decision and pair[2]:
                    f_n_dyn = f_n_dyn + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_dyn / positive) - (f_p_dyn / negative)

        eps_static = scipy.optimize.bisect(f_s, -1, 1)
        eps_dynamic = scipy.optimize.bisect(f_d, -1, 1)

        # --- calculate the eer for dynamic and static ---
        f_p_static = 0
        f_n_static = 0
        score_static = 0
        positive = 0
        negative = 0
        for pair in static_pairs:
            decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_static
            if (decision and pair[2]) or (not decision and not pair[2]):
                score_static = score_static + 1
            elif decision and not pair[2]:
                f_p_static = f_p_static + 1
            elif not decision and pair[2]:
                f_n_static = f_n_static + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_static = (f_p_static) / negative

        f_p_dyn = 0
        f_n_dyn = 0
        positive = 0
        negative = 0
        for pair in dynamic_pairs:
            decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_dynamic
            if decision and not pair[2]:
                f_p_dyn = f_p_dyn + 1
            elif not decision and pair[2]:
                f_n_dyn = f_n_dyn + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_dynamic = (f_p_dyn) / negative

        if run:
            run['eer_static'].log(eer_static)
            run['eer_dynamic'].log(eer_dynamic)
            print(
                'err_static {},  err_dynamic {},  final_eps_dyn {}, final_eps_st {}'.format(
                    eer_static,
                    eer_dynamic,
                    eps_dynamic,
                    eps_static))
        else:
            print(
                'err_static {},  err_dynamic {},  final_eps_dyn {}, final_eps_st {}'.format(
                    eer_static,
                    eer_dynamic,
                    eps_dynamic,
                    eps_static))

        return eer_static, eer_dynamic


def voice_verification_dbse_mean(dbse, spectrogram, test_loader, run):
    for epoch in range(opt.niter):

        print("Epoch", epoch)
        dbse.eval()
        # mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        # mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        # pred1_all, pred2_all, label2_all = list(), list(), list()
        # label_gt = list()

        dataset = list(test_loader.dataset)
        wavs, chunck_nums = audio_path_to_data_dbse([w['wav'] for w in dataset])

        X = wavs
        # encode
        X = X.cuda()
        X = spectrogram(X)
        X = X.permute(0, 2, 1)
        f_mean, f_logvar, f_post, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_frame_x, recon_seq_x = dbse(
            X)  # pred

        n = 0
        f_means_all = []
        for num in chunck_nums:
            f_means_all.append(torch.mean(f_mean[n:n + num], dim=0))
            n += num
        f_post_mean = torch.stack(f_means_all)

        n = 0
        z_means_all = []
        for num in chunck_nums:
            z_means_all.append(torch.mean(torch.mean(z_post[n:n + num], dim=0), dim=0))
            n += num
        z_post_mean = torch.stack(z_means_all)

        # --- create pairs of verifications and expected output ---
        index_comb = list(itertools.combinations(range(len(dataset)), 2))

        static_pairs = []
        dynamic_pairs = []
        dataset_list = list(dataset)
        # take the frames for each pair of samples
        for comb in index_comb:
            static_pairs.append(
                [f_post_mean[comb[0]], f_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])
            dynamic_pairs.append(
                [z_post_mean[comb[0]], z_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])

        # --- def for binary search ---
        def f_s(epsilon):
            f_p_static = 0
            f_n_static = 0
            positive = 0
            negative = 0
            for pair in static_pairs:
                decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_static = f_p_static + 1
                elif not decision and pair[2]:
                    f_n_static = f_n_static + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_static / positive) - (f_p_static / negative)

        def f_d(epsilon):
            f_p_dyn = 0
            f_n_dyn = 0
            positive = 0
            negative = 0
            for pair in dynamic_pairs:
                decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_dyn = f_p_dyn + 1
                elif not decision and pair[2]:
                    f_n_dyn = f_n_dyn + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_dyn / positive) - (f_p_dyn / negative)

        eps_static = scipy.optimize.bisect(f_s, -1, 1)
        eps_dynamic = scipy.optimize.bisect(f_d, -1, 1)

        # --- calculate the eer for dynamic and static ---
        f_p_static = 0
        f_n_static = 0
        score_static = 0
        positive = 0
        negative = 0
        for pair in static_pairs:
            decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_static
            if (decision and pair[2]) or (not decision and not pair[2]):
                score_static = score_static + 1
            elif decision and not pair[2]:
                f_p_static = f_p_static + 1
            elif not decision and pair[2]:
                f_n_static = f_n_static + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_static = (f_p_static) / negative

        f_p_dyn = 0
        f_n_dyn = 0
        positive = 0
        negative = 0
        for pair in dynamic_pairs:
            decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_dynamic
            if decision and not pair[2]:
                f_p_dyn = f_p_dyn + 1
            elif not decision and pair[2]:
                f_n_dyn = f_n_dyn + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_dynamic = (f_p_dyn) / negative

        if run:
            run['eer_static'].log(eer_static)
            run['eer_dynamic'].log(eer_dynamic)
            print(
                'err_static {},  err_dynamic {},  final_eps_dyn {}, final_eps_st {}'.format(
                    eer_static,
                    eer_dynamic,
                    eps_dynamic,
                    eps_static))
        else:
            print(
                'err_static {},  err_dynamic {},  final_eps_dyn {}, final_eps_st {}'.format(
                    eer_static,
                    eer_dynamic,
                    eps_dynamic,
                    eps_static))

        return eer_static, eer_dynamic


def voice_verification(dbse, spectrogram, test_loader, run):
    for epoch in range(opt.niter):

        print("Epoch", epoch)
        dbse.eval()
        # mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        # mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        # pred1_all, pred2_all, label2_all = list(), list(), list()
        # label_gt = list()

        dataset = list(test_loader.dataset)
        wavs, lens = audio_path_to_data([w['wav'] for w in dataset])

        X = wavs
        # encode
        X = X.cuda()
        X = spectrogram(X)
        X = X.permute(0, 2, 1)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
        recon_x, _ = dbse(X)

        z_post_mean = []
        # get rid of padding encoding:
        for i in range(z_post.shape[0]):
            z_post_mean.append(torch.mean(z_post[i, :lens[i], :], dim=0))

        f_post_mean = f_post
        z_post_mean = torch.stack(z_post_mean)  # DEBUG AND SEE IF IT ACTUALLY MEAN VALUE OF THE SEQUENCE DIM

        # --- create pairs of verifications and expected output ---
        index_comb = list(itertools.combinations(range(len(dataset)), 2))

        static_pairs = []
        dynamic_pairs = []
        dataset_list = list(dataset)
        # take the frames for each pair of samples
        for comb in index_comb:
            static_pairs.append(
                [f_post_mean[comb[0]], f_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])
            dynamic_pairs.append(
                [z_post_mean[comb[0]], z_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])

        # --- def for binary search ---
        def f_s(epsilon):
            f_p_static = 0
            f_n_static = 0
            positive = 0
            negative = 0
            for pair in static_pairs:
                decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_static = f_p_static + 1
                elif not decision and pair[2]:
                    f_n_static = f_n_static + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_static / positive) - (f_p_static / negative)

        def f_d(epsilon):
            f_p_dyn = 0
            f_n_dyn = 0
            positive = 0
            negative = 0
            for pair in dynamic_pairs:
                decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_dyn = f_p_dyn + 1
                elif not decision and pair[2]:
                    f_n_dyn = f_n_dyn + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_dyn / positive) - (f_p_dyn / negative)

        eps_static = scipy.optimize.bisect(f_s, -1, 1)
        eps_dynamic = scipy.optimize.bisect(f_d, -1, 1)

        # --- calculate the eer for dynamic and static ---
        f_p_static = 0
        f_n_static = 0
        score_static = 0
        positive = 0
        negative = 0
        for pair in static_pairs:
            decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_static
            if (decision and pair[2]) or (not decision and not pair[2]):
                score_static = score_static + 1
            elif decision and not pair[2]:
                f_p_static = f_p_static + 1
            elif not decision and pair[2]:
                f_n_static = f_n_static + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_static = (f_p_static) / negative

        f_p_dyn = 0
        f_n_dyn = 0
        positive = 0
        negative = 0
        for pair in dynamic_pairs:
            decision = torch.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_dynamic
            if decision and not pair[2]:
                f_p_dyn = f_p_dyn + 1
            elif not decision and pair[2]:
                f_n_dyn = f_n_dyn + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_dynamic = (f_p_dyn) / negative

        if run:
            run['eer_static'].log(eer_static)
            run['eer_dynamic'].log(eer_dynamic)
        else:
            print(
                'err_static {},  err_dynamic {},  final_eps_dyn {}, final_eps_st {}'.format(
                    eer_static,
                    eer_dynamic,
                    eps_dynamic,
                    eps_static))

        return eer_static, eer_dynamic


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # set the data folder
    data_folder = hparams.data_folder
    # get train annotations
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams.train_annotation,
        replacements={"data_root": data_folder},
    )

    # we sort training data to speed up training and get better results.
    train_data = train_data.filtered_sorted(sort_key="duration")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams.valid_annotation,
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams.test_annotation,
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder
    # lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    # special_labels = {
    #     "bos_label": hparams["bos_index"],
    #     "eos_label": hparams["eos_index"],
    #     "blank_label": hparams["blank_index"],
    # }
    # if not os.path.exists(hparams["save_folder"]):
    #     os.mkdir(hparams["save_folder"])
    # label_encoder.load_or_create(
    #     path=lab_enc_file,
    #     from_didatasets=[train_data],
    #     output_key="phn_list",
    #     special_labels=special_labels,
    #     sequence_input=True,
    # )

    # 4. Set output:
    # sb.dataio.dataset.set_output_keys(
    #     datasets,
    #     ["id", "sig", "phn_encoded", "phn_encoded_eos", "phn_encoded_bos"],
    # )

    return train_data, valid_data, test_data, label_encoder


def load_dataset(opt):
    from load_timit import prepare_timit
    from speechbrain.utils.distributed import run_on_main

    # set configurations to enable data loading
    opt.data_folder = opt.dataset_path
    opt.train_annotation = 'your train path'
    opt.valid_annotation = 'your valid path'
    opt.test_annotation = 'your test path'

    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": opt.dataset_path,
            "save_json_train": opt.train_annotation,
            "save_json_valid": opt.valid_annotation,
            "save_json_test": opt.test_annotation,
            "skip_prep": False,
        },
    )

    train_data, valid_data, test_data, label_encoder = dataio_prep(opt)

    # join train and valid
    train_data.data.update(valid_data.data)

    return train_data, test_data


# --------- training funtions ------------------------------------
def train(x, model, optimizer, opt, mode="train"):
    if mode == "train":
        model.zero_grad()

    if isinstance(x, list):
        batch_size = x[0].size(0)
    else:
        batch_size = x.size(0)

    f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_frame_x, recon_seq_x = model(x)  # pred

    if opt.loss_recon == 'L2':  # True branch
        l_recon = F.mse_loss(recon_seq_x, x, reduction='sum')
    else:
        l_recon = torch.abs(recon_seq_x - x).sum()

    if opt.loss_recon == 'L2':  # True branch
        l_recon_frame = F.mse_loss(recon_frame_x, x, reduction='sum')
    else:
        l_recon_frame = torch.abs(recon_frame_x - x[:, 0]).sum()

    f_mean = f_mean.view((-1, f_mean.shape[-1]))
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    l_recon, l_recon_frame, kld_f, kld_z = l_recon / batch_size, l_recon_frame / batch_size, kld_f / batch_size, kld_z / batch_size

    loss = l_recon * opt.weight_rec + l_recon_frame * opt.weight_rec_frame + kld_f * opt.weight_f + kld_z * opt.weight_z

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return [i.data.cpu().numpy() for i in [l_recon, l_recon_frame, kld_f, kld_z]] + [recon_seq_x]


def log_to_neptune(run, neptune_log_dic):
    for key, value in neptune_log_dic.items():
        run[key].log(value)


def main(hyperparameters):
    opt.z_dim = hyperparameters['l_dim']
    opt.f_dim = hyperparameters['l_dim']
    opt.lr = hyperparameters['lr']
    opt.weight_rec = hyperparameters['weight_rec']
    opt.weight_rec_frame = hyperparameters['weight_rec_frame']
    opt.batch_size = hyperparameters['batch_size']
    opt.weight_f = hyperparameters['weight_f']
    opt.weight_z = hyperparameters['weight_z']

    eer_static = None
    name = 'DBSE_TIMIT_latent_epoch-{}_bs-{}_decoder={}-rnn_size={}-g_dim={}-f_dim={}-z_dim={}-lr={}' \
           '-weight:kl_f={}-kl_z={}-c_aug={}-m_aug={}-{}-sche_{}-{}-c_lval={}--tag={}'.format(
        opt.nEpoch, opt.batch_size, opt.decoder, opt.rnn_size, opt.g_dim, opt.f_dim,
        opt.z_dim, opt.lr,
        opt.weight_f, opt.weight_z, opt.weight_c_aug, opt.weight_m_aug,
        opt.loss_recon, opt.sche, opt.c_loss, opt.c_floss, opt.tag)

    spectrogram = torchaudio.transforms.Spectrogram(win_length=opt.w_len, hop_length=opt.h_len, power=opt.power).to('cuda')

    run = None
    if opt.neptune:
        run = neptune.init_run(
            project="Your Neptune Path",
            api_token="Your Neptune API Token",
        )  # your credentials
        run['config/hyperparameters'] = vars(opt)
        run['config/name'] = name
        run['sys/tags'].add([opt.tag, "b-lstm-linear"])

    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    log = os.path.join(opt.log_dir, 'log.txt')

    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    print_log("Random Seed: {}".format(opt.seed), log)

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)

    # control the sequence sample
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    print_log('Running parameters:')

    # ---------------- optimizers ----------------
    opt.optimizer = optim.Adam
    dbse_model = DBSE(opt)
    dbse_model.apply(utils.init_weights)
    optimizer = opt.optimizer(dbse_model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    if opt.sche == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch + 1) // 2, T_mult=1)
    elif opt.sche == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.nEpoch // 2, gamma=0.5)
    elif opt.sche == "const":
        scheduler = None
    else:
        raise ValueError('unknown scheduler')

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.device_count() > 1:
        dbse_model = nn.DataParallel(dbse_model)

    dbse_model = dbse_model.cuda()

    # --------- load a dataset ------------------------------------
    train_data, test_data = load_dataset(opt)
    train_loader = DataLoader(list(train_data.data.values()),
                              num_workers=4,
                              batch_size=opt.batch_size,  # 128
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(list(test_data.data.values()),
                             num_workers=4,
                             batch_size=opt.batch_size,  # 128
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)
    opt.dataset_size = len(train_data)

    epoch_loss = Loss()

    # --------- training loop ------------------------------------
    for epoch in range(opt.nEpoch):
        if epoch and scheduler is not None:
            scheduler.step()

        if epoch == opt.c_loss:
            print('start contrastive loss computation')
            opt.weight_c_aug, opt.weight_m_aug = opt.c_floss, opt.c_floss

        dbse_model.train()
        epoch_loss.reset()

        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(maxval=len(train_loader)).start()
        for i, data in enumerate(train_loader):
            progress.update(i + 1)
            x, _ = audio_path_to_data_dbse(data['wav'])
            x = x.cuda()

            # transform with STFT and a spectrogram function to create learning features
            x = spectrogram(x)
            # X dimensions = b x f x t where f is the
            # transfer x to be - b x t x f

            x = x.permute(0, 2, 1)

            # train frame_predictor
            recon_seq_train, recon_frame, kld_f, kld_z, recon_train_exp = train(x, dbse_model, optimizer, opt)

            lr = optimizer.param_groups[0]['lr']
            if opt.neptune:
                log_to_neptune(run, {'train/lr': lr,
                                     'train/mse_seq': recon_seq_train.item(),
                                     'train/mse_frame': recon_frame.item(),
                                     'train/kld_f': kld_f.item(),
                                     'train/kld_z': kld_z.item()})

                epoch_loss.update(recon_seq_train, kld_f, kld_z, 0, 0)

        progress.finish()
        utils.clear_progressbar()
        avg_loss = epoch_loss.avg()
        print_log('[%02d] recon: %.2f | kld_f: %.2f | kld_z: %.2f | con_loss_c: %.5f |'
                  ' con_loss_m: %.5f | lr: %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2],
                                                    avg_loss[3], avg_loss[4], lr), log)

        if epoch % opt.evl_interval == 0 or epoch == opt.nEpoch - 1:
            dbse_model.eval()
            # save the model
            net2save = dbse_model.module if torch.cuda.device_count() > 1 else dbse_model
            torch.save({
                'model': net2save.state_dict(),
                'optimizer': optimizer.state_dict()},
                '%s/model%d.pth' % (opt.log_dir, opt.nEpoch))

        if epoch == opt.nEpoch - 1 or epoch % 5 == 0:
            val_mse_seq = val_mse_frame = val_kld_f = val_kld_z = val_c_loss = val_m_loss = val_mi_xs = val_mi_fz = val_mi_xz = 0.
            for i, data in enumerate(test_loader):
                x_test, _ = audio_path_to_data_dbse(data['wav'])
                x_test = x_test.cuda()

                # transform with STFT and a spectrogram function to create learning features
                x_test = spectrogram(x_test)
                # X dimensions = b x f x t where f is the

                # transfer x to be - b x t x f
                x_test = x_test.permute(0, 2, 1)

                with torch.no_grad():
                    recon_seq_test, recon_frame, kld_f, kld_z, recon_test_exp = train(
                        x_test, dbse_model,
                        optimizer,
                        opt,
                        mode="val")

                val_mse_seq += recon_seq_test
                val_mse_frame += recon_frame
                val_kld_f += kld_f
                val_kld_z += kld_z

            n_batch = len(test_loader)

            if opt.neptune:
                log_to_neptune(run, {'test/mse_seq': val_mse_seq.item() / n_batch,
                                     'test/mse_frame': val_mse_frame.item() / n_batch,
                                     'test/kld_f': val_kld_f.item() / n_batch,
                                     'test/kld_z': val_kld_z.item() / n_batch})

        if epoch % 10 == 0 and epoch > 20:
            dbse_model.eval()
            eer_static, _ =  voice_verification_dbse_mean(dbse_model, spectrogram, test_loader, run)
            if epoch >= 50 and eer_static > 0.1:
                break
            if epoch >= 100 and eer_static > 0.06:
                break

        if epoch % 30 == 0 and epoch > 0:
            dbse_model.eval()
            # voice_verification_cdsvae(cdsvae, spectrogram, test_loader, run)
            if opt.neptune:
                imshow_spec(specs=[recon_train_exp, x], titles=['rec', 'orig'], neptune=run,
                            file_name='train/orig_vs_rec')
                imshow_spec(specs=[recon_test_exp, x_test], titles=['rec', 'orig'], neptune=run,
                            file_name='test/orig_vs_rec')

                # swap_timit(cdsvae, x_test, chunck_nums, run, epoch)

                from_spectrogram_to_wav(x_test[0:22].reshape(-1, 201), run, 'real')
                from_spectrogram_to_wav(recon_test_exp[0:22].reshape(-1, 201), run, 'rec_epc{}'.format(epoch))

    if opt.neptune:
        run.stop()
    print("Training is complete")

    return eer_static


def swap_timit(dbse, x, chunks_num, run, epoch):
    f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
    recon_x, _ = dbse(x[0:sum(chunks_num[0:2])])

    f1_mean = f_mean[:chunks_num[0]].mean(dim=0)
    f2_mean = f_mean[chunks_num[0]:].mean(dim=0)
    z1_mean = z_mean_post[:chunks_num[0]].reshape(1, -1, z_mean_post[:chunks_num[0]].shape[2])
    z2_mean = z_mean_post[chunks_num[0]:].reshape(1, -1, z_mean_post[chunks_num[0]:].shape[2])

    f1_expand = f1_mean.unsqueeze(0).unsqueeze(0).expand(-1, z2_mean.shape[1], f1_mean.shape[-1])
    f2_expand = f2_mean.unsqueeze(0).unsqueeze(0).expand(-1, z1_mean.shape[1], f2_mean.shape[-1])

    rec_s1d2 = dbse.decoder(torch.cat([z2_mean, f1_expand], dim=2))
    rec_s2d1 = dbse.decoder(torch.cat([z1_mean, f2_expand], dim=2))

    from_spectrogram_to_wav(rec_s1d2.reshape(-1, 201), run, 'rec_s1d2_{}'.format(epoch))
    from_spectrogram_to_wav(rec_s2d1.reshape(-1, 201), run, 'rec_s2d1_{}'.format(epoch))
    from_spectrogram_to_wav(recon_x[:chunks_num[0]].reshape(-1, 201), run, 'recon_x_1_{}'.format(epoch))
    from_spectrogram_to_wav(recon_x[chunks_num[0]:].reshape(-1, 201), run, 'recon_x_2_{}'.format(epoch))

    rec_s10 = dbse.decoder(torch.cat([z1_mean, torch.zeros(f2_expand.shape).to(f1_expand.device)], dim=2))
    rec_d10 = dbse.decoder(torch.cat([torch.zeros(z2_mean.shape).to(z2_mean.device), f1_expand], dim=2))

    from_spectrogram_to_wav(rec_s10.reshape(-1, 201), run, 'rec_s10{}'.format(epoch))
    from_spectrogram_to_wav(rec_d10.reshape(-1, 201), run, 'rec_d10{}'.format(epoch))


def imshow_spec(specs, titles, neptune, file_name):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(specs[0][0:20].reshape(-1, 201).detach().cpu().permute(1, 0), origin='lower')
    ax1.title.set_text(titles[0])
    ax2.imshow(specs[1][0:20].reshape(-1, 201).detach().cpu().permute(1, 0), origin='lower')
    ax2.title.set_text(titles[1])
    neptune[file_name].log(f)


def from_spectrogram_to_time_domain(data, n_fft=400, win_len=320, hop_len=165, power=1):
    griffin_lim = ts.GriffinLim(n_iter=50, n_fft=n_fft, win_length=win_len, hop_length=hop_len, power=power).to(
        data.device)
    x = griffin_lim(data.permute(1, 0))

    return x


def from_time_domain_to_wav(data, run, file_name, sample_rate=16000):
    write_audio('./{}.wav'.format(file_name), data.cpu(), sample_rate)
    run["audio/{}".format(file_name)] = neptune.types.File('./{}.wav'.format(file_name))

    return 0


def from_spectrogram_to_wav(data, run, file_name):
    x = from_spectrogram_to_time_domain(data)
    from_time_domain_to_wav(x, run, file_name)

    return 0


# X, X, 64, 64, 3 -> # X, X, 3, 64, 64
def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)


def get_batch(train_loader):
    while True:
        for sequence in train_loader:
            yield sequence


def print_log(print_string, log=None, verbose=True):
    if verbose:
        print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()


class Loss(object):
    def __init__(self):
        self.reset()

    def update(self, recon, kld_f, kld_z, con_loss_c, con_loss_m):
        self.recon.append(recon)
        self.kld_f.append(kld_f)
        self.kld_z.append(kld_z)
        self.con_loss_c.append(con_loss_c)
        self.con_loss_m.append(con_loss_m)

    def reset(self):
        self.recon = []
        self.kld_f = []
        self.kld_z = []
        self.con_loss_c = []
        self.con_loss_m = []

    def avg(self):
        return [np.asarray(i).mean() for i in
                [self.recon, self.kld_f, self.kld_z, self.con_loss_c, self.con_loss_m]]



def get_experiment_space():
    space = {  # Architecture parameters
        'model': 'dbse_timit',
        'mode': 'simple',
        'l_dim': hp.choice('l_dim', [128, 120, 110, 100, 90, 80, 70]),
        'lr': hp.choice('lr', [0.001]),
        'weight_rec': hp.choice('weight_rec_seq', [ 150., 160., 140., 155., 170., 185., 200.]),
        'weight_f':  hp.choice('weight_f', [1.]),
        'weight_z': hp.choice('weight_z', [1.]),
        'weight_rec_frame': hp.choice('weight_rec_frame', [0.01, 0, 0.001]),
        'batch_size': hp.choice('batch_size', [32, 16, 10])}

    return space

def objective(hyperparameters):
    # The objective function is what we minimize.
    # We want to maximize a_action, so we minimize negative a_action.
    return main(hyperparameters)

if __name__ == '__main__':
    trials = Trials()
    best = fmin(fn=objective, space=get_experiment_space(), algo=tpe.suggest, max_evals=20)

