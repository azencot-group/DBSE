import torch
import numpy as np
import os, sys
import random
import torch.nn as nn
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clear_progressbar():
    """
    Clears the progress bar in the console output.

    :return: None
    """
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


def init_weights(model):
    """
    Initializes the weights of the model's layers.

    :param model: The model whose weights are to be initialized.
    :return: None
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
    Calculates the entropy H(Y) from conditional probabilities.

    :param p_yx: Conditional probability distribution p(y|x).
    :param eps: Small value to prevent log(0).
    :return: Entropy H(Y).
    """
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h


def entropy_Hyx(p, eps=1E-16):
    """
    Calculates the average entropy H(Y|X).

    :param p: Joint probability distribution p(y,x).
    :param eps: Small value to prevent log(0).
    :return: Average conditional entropy H(Y|X).
    """
    sum_h = (p * np.log(p + eps)).sum(axis=1)
    # average over video
    avg_h = np.mean(sum_h) * (-1)
    return avg_h


def inception_score(p_yx, eps=1E-16):
    """
    Computes the Inception Score based on conditional probabilities.

    :param p_yx: Conditional probability distribution p(y|x).
    :param eps: Small value to prevent log(0).
    :return: Inception Score.
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
    Calculates the Kullback-Leibler divergence between two distributions.

    :param P: Probability distribution P.
    :param Q: Probability distribution Q.
    :param eps: Small value to prevent log(0).
    :return: Average KL divergence.
    """
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over video
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d


def define_scheduler(opt, optimizer):
    """
    Defines a learning rate scheduler based on provided options.

    :param opt: Options or configuration settings.
    :param optimizer: Optimizer for which the scheduler is defined.
    :return: Configured learning rate scheduler.
    """
    schedulers = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4,
                                                                       T_0=(opt.nEpoch + 1) // 2, T_mult=1),
        "step": torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.nEpoch // 2, gamma=0.5),
        "const": None,
    }

    try:
        scheduler = schedulers[opt.sche]
    except KeyError:
        raise ValueError('unknown scheduler')

    return scheduler


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


def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    """
    Generates a cosine range for learning rate scheduling over a specified number of epochs.

    :param start: Starting value of the range.
    :param stop: Ending value of the range.
    :param n_epoch: Total number of epochs.
    :param n_cycle: Number of cycles for the cosine wave.
    :param ratio: Ratio to determine the step size.
    :return: Array representing the cosine range.
    """
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L
