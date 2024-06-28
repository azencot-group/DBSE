import torch
import numpy as np
import os, sys
import random
import torch.nn as nn
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchvision
from matplotlib import pyplot as plt
from pynvml import *
import socket

# X, X, 64, 64, 3 -> # X, X, 3, 64, 64
def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)

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


# Schedulers are used to adjust the learning rate during training. They modify the learning rate based on the number of epochs, or steps, improving the model's ability to find optimal solutions.
def define_scheduler(opt, optimizer):
    schedulers = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch + 1) // 2, T_mult=1),
        "step": torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.nEpoch // 2, gamma=0.5),
        "const": None,
    }

    try:
        scheduler = schedulers[opt.sche]
    except KeyError:
        raise ValueError('unknown scheduler')

    return scheduler


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


def use_multiple_gpus_if_possible(model, log_file, opt):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print_log(f"Running is using {num_gpus} GPUs!", log_file)
        opt.device = 'cuda'
        model = nn.DataParallel(model)
    return model


def get_gpu_unique_id():
    # Initialize NVML
    nvmlInit()

    # Get the handle for the first GPU device
    handle = nvmlDeviceGetHandleByIndex(0)

    # Get the GPU UUID
    try:
        uuid = nvmlDeviceGetUUID(handle)
        return uuid
    except Exception:
        return False

    # Shutdown NV ML
    nvmlShutdown()


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command doesn't actually connect to the external server, it just gets your IP.
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = False
    finally:
        s.close()
    return IP


def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
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
