import random
import utils
import numpy as np
import os
import argparse

import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from sprites_utils_train import *
from dbse_model import DBSE, classifier_Sprite_all

from utils import get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--nEpoch', default=1000, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')
parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')
parser.add_argument('--neptune', default=False, type=bool, help='activate neptune tracking')

parser.add_argument('--dataset', default='Sprite', type=str, help='dataset to train')
parser.add_argument("--dataset_path", default='/cs/cs_groups/azencot_group/datasets/SPRITES_ICML/datasetICML/')
parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--decoder', default='ConvT', type=str, help='Upsampling+Conv or Transpose Conv: Conv or ConvT')

parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=64, type=int, help='dimensionality of hidden layer')
parser.add_argument('--f_dim', default=256, type=int, help='dim of f')
parser.add_argument('--z_dim', default=256, type=int, help='dimensionality of z_t')
parser.add_argument('--g_dim', default=256, type=int,
                    help='dimensionality of encoder output vector and decoder input vector')

parser.add_argument('--loss_recon', default='L2', type=str, help='reconstruction loss: L1, L2')
parser.add_argument('--note', default='LogNCELoss', type=str, help='appx note')
parser.add_argument('--weight_f', default=1, type=float, help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', default=1, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_rec_seq', default=5, type=float, help='weighting on reconstruction')
parser.add_argument('--weight_rec_frame', default=1, type=float, help='weighting on reconstruction')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--sche', default='const', type=str, help='scheduler')

parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
parser.add_argument('--niter', type=int, default=5, help='number of runs for testing')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

mse_loss = nn.MSELoss().cuda()

if opt.seed is None:
    opt.seed = random.randint(1, 10000)

# control the sequence sample
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)


# load model
cdsvae = DBSE(opt).cuda()
cdsvae.load_state_dict(torch.load('/home/arbivid/sprites_no_mi/logs/Sprite/NA_Sprite_epoch-2000_bs-128_decoder=ConvT64x64-rnn_size=64-g_dim=256-f_dim=256-z_dim=256-lr=0.002-weight:kl_f=1-kl_z=1-0-L2-sche_const-note=LogNCELoss/model2000.pth')['model'])
cdsvae.eval()

# load classifier for testing during train
old_rnn_size = opt.rnn_size
old_g_dim = opt.g_dim
opt.rnn_size = 256
opt.g_dim = 128
classifier = classifier_Sprite_all(opt)
opt.resume = '/cs/cs_groups/azencot_group/mutual_information_disentanglement/classifiers/sprite_judge.tar'
loaded_dict = torch.load(opt.resume)
classifier.load_state_dict(loaded_dict['state_dict'])
classifier = classifier.cuda().eval()
opt.rnn_size = old_rnn_size
opt.g_dim = old_g_dim

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)
N, seq_len, dim1, dim2, n_c = train_data.data.shape
train_loader = DataLoader(train_data,
                          num_workers=4,
                          batch_size=opt.batch_size,  # 128
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=4,
                         batch_size=opt.batch_size,  # 128
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)
test_video_enumerator = get_batch(test_loader)
opt.dataset_size = len(train_data)

opt.type_gt = 'action'
check_cls(opt, cdsvae, classifier, test_loader, None)
opt.type_gt = 'aaction'
check_cls(opt, cdsvae, classifier, test_loader, None)

