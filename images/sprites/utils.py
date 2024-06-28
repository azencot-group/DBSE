import random

import torch
import numpy as np

import torchvision
from matplotlib import pyplot as plt


# matplotlib.use('agg')


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


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


import torch.nn as nn


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def entropy_Hy(p_yx, eps=1E-16):
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h


def entropy_Hyx(p, eps=1E-16):
    sum_h = (p * np.log(p + eps)).sum(axis=1)
    # average over images
    avg_h = np.mean(sum_h) * (-1)
    return avg_h


def inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score


def KL_divergence(P, Q, eps=1E-16):
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d


def imshow_seq_simple(data, neptune=None, file_name="", plot=False):
    # npimg = data.numpy()
    npimg = data
    np.transpose(torchvision.utils.make_grid(torch.tensor(npimg)), (1, 2, 0))
    frame1 = plt.gca()
    plt.imshow(np.transpose(torchvision.utils.make_grid(torch.tensor(npimg)), (1, 2, 0)))
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    # plt.savefig('./results/mug_simclr_examples.pdf', transparent=True, bbox_inches='tight', pad_inches=0,
    #             dpi=300)
    if neptune is not None:
        fig = plt.figure(1, (2, 2))
        neptune[file_name].log(fig)
    elif plot:
        plt.show()
    else:
        return plt.figure(1, (2, 2))


def load_dataset(opt):
    if opt.dataset == 'Sprite':
        path = opt.dataset_path
        from sprites_dataloader import Sprite

        with open(path + 'sprites_X_train.npy', 'rb') as f:
            X_train = np.load(f)
        with open(path + 'sprites_X_test.npy', 'rb') as f:
            X_test = np.load(f)
        with open(path + 'sprites_A_train.npy', 'rb') as f:
            A_train = np.load(f)
        with open(path + 'sprites_A_test.npy', 'rb') as f:
            A_test = np.load(f)
        with open(path + 'sprites_D_train.npy', 'rb') as f:
            D_train = np.load(f)
        with open(path + 'sprites_D_test.npy', 'rb') as f:
            D_test = np.load(f)

        train_data = Sprite(data=X_train, A_label=A_train, D_label=D_train)
        test_data = Sprite(data=X_test, A_label=A_test, D_label=D_test)

    else:
        raise ValueError('no implementation of dataset {}'.format(opt.dataset))

    return train_data, test_data


def define_seed(opt):
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
        print("Random Seed (automatically generated): ", opt.seed)
    else:
        print("Random Seed (user defined): ", opt.seed)

    # Control the sequence sample
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
