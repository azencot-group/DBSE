import random

import torch
import numpy as np


def reorder(sequence):
    """
    Reorders the dimensions of the input sequence tensor.

    :param sequence: A tensor of shape (batch_size, time, channels, height, width).
    :return: A tensor with reordered dimensions (batch_size, time, height, width, channels).
    """
    return sequence.permute(0, 1, 4, 2, 3)


def clear_progressbar():
    """
    Clears the progress bar in the terminal by moving the cursor up and deleting previous lines.
    This is useful for refreshing the progress display.

    :return: None.
    """
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


import torch.nn as nn


def init_weights(model):
    """
    Initializes the weights of the model layers using standard initialization techniques.

    :param model: A neural network model where layers such as Conv2d, BatchNorm2d, and Linear will be initialized.
    :return: None.
    """
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
    """
    Computes the entropy H(y) given the conditional probability distribution p(y|x).

    :param p_yx: A 2D array of conditional probabilities p(y|x) of shape (N, C), where N is the number of samples and C is the number of classes.
    :param eps: A small value to avoid logarithm of zero (default: 1E-16).
    :return: A scalar representing the entropy H(y).
    """
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h


def entropy_Hyx(p, eps=1E-16):
    """
    Computes the conditional entropy H(y|x) for each sample in the distribution.

    :param p: A 2D array of conditional probabilities p(y|x) of shape (N, C), where N is the number of samples and C is the number of classes.
    :param eps: A small value to avoid logarithm of zero (default: 1E-16).
    :return: A scalar representing the conditional entropy H(y|x).
    """
    sum_h = (p * np.log(p + eps)).sum(axis=1)
    # average over video
    avg_h = np.mean(sum_h) * (-1)
    return avg_h


def inception_score(p_yx, eps=1E-16):
    """
    Computes the Inception Score (IS) from the conditional probability distribution p(y|x).

    :param p_yx: A 2D array of conditional probabilities p(y|x) of shape (N, C), where N is the number of samples and C is the number of classes.
    :param eps: A small value to avoid logarithm of zero (default: 1E-16).
    :return: A scalar representing the Inception Score.
    """
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over video
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score


def KL_divergence(P, Q, eps=1E-16):
    """
    Computes the Kullback-Leibler (KL) divergence between two distributions P and Q.

    :param P: A 2D array representing the true probability distribution of shape (N, C).
    :param Q: A 2D array representing the approximate probability distribution of shape (N, C).
    :param eps: A small value to avoid logarithm of zero (default: 1E-16).
    :return: A scalar representing the KL divergence between P and Q.
    """
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over video
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d


def load_dataset(opt):
    """
    Loads a dataset based on the provided option settings.

    :param opt: An object with dataset configuration options. The attribute `dataset` specifies the dataset type, and `dataset_path` specifies the path to the dataset files.
    :return: A tuple (train_data, test_data), where each is a dataset object.
    :raises ValueError: If the dataset is not implemented.
    """
    if opt.dataset == 'Sprite':
        path = opt.dataset_path
        if path is None:
            raise ValueError("A 'dataset_path' must be provided by the user.")
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
    """
    Defines and sets the random seed for various modules to ensure reproducibility.

    :param opt: An object containing a seed attribute. If not provided, a random seed will be generated.
    :return: None.
    """
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
