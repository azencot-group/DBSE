import json
import random
import functools
import PIL
import utils
import progressbar
import numpy as np
import os
import argparse
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from dbse_model import DBSE, classifier_Sprite_all

import neptune.new as neptune
from utils import print_log, get_batch, reorder
from sprites_utils_train import check_cls

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--nEpoch', default=2000, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval', default=20, type=int, help='evaluate every n epoch')
parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')
parser.add_argument('--neptune', default=True, type=bool, help='activate neptune tracking')

parser.add_argument('--dataset', default='Sprite', type=str, help='dataset to train')
parser.add_argument("--dataset_path", default='/cs/cs_groups/azencot_group/datasets/SPRITES_ICML/datasetICML/')
parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
parser.add_argument('--channels', default=3, type=int, help='number of channels in video')
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


# --------- training funtions ------------------------------------
def train(x, model, optimizer, opt, mode="train"):
    if mode == "train":
        model.zero_grad()

    batch_size = x.size(0)

    f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_frame_x, \
    recon_seq_x = model(x)  # pred

    if opt.loss_recon == 'L2':  # True branch
        l_recon = F.mse_loss(recon_seq_x, x, reduction='sum')
    else:
        l_recon = torch.abs(recon_seq_x - x).sum()

    if opt.loss_recon == 'L2':  # True branch
        l_recon_frame = F.mse_loss(recon_frame_x, x[:, 0], reduction='sum')
    else:
        l_recon_frame = torch.abs(recon_frame_x - x[:, 0]).sum()

    f_mean = f_mean.view((-1, f_mean.shape[-1]))  # [128, 256]
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar)  # [128, 8, 32]
    z_prior_var = torch.exp(z_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    l_recon, l_recon_frame, kld_f, kld_z = l_recon / batch_size, l_recon_frame / batch_size, kld_f / batch_size, \
                                           kld_z / batch_size

    loss = l_recon * opt.weight_rec_seq + l_recon_frame * opt.weight_rec_frame + kld_f * opt.weight_f + kld_z * opt.weight_z

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return [i.data.cpu().numpy() for i in [l_recon, l_recon_frame, kld_f, kld_z]]


def main(opt):
    # opt.g_dim = hyperparameters['g_dim']
    # opt.rnn_size = hyperparameters['rnn_size']
    # opt.f_dim = hyperparameters['l_dim']
    # opt.z_dim = hyperparameters['l_dim']
    # opt.weight_rec_seq = hyperparameters['weight_rec_seq']
    # opt.weight_rec_frame = hyperparameters['weight_rec_frame']
    a_action = 0

    run = None
    opt.rng = 1234
    if opt.neptune:
        run = neptune.init_run(
            project="azencot-group/No-Mi-Sprites",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjODljNGI3NS0yOWYyLTRhN2QtOWVmYS1iMTE4ODAxYWM5NmQifQ==",
        )
        run['config/hyperparameters'] = vars(opt)

    name = 'NA_Sprite_epoch-{}_bs-{}_decoder={}{}x{}-rnn_size={}-g_dim={}-f_dim={}-z_dim={}-lr={}' \
           '-weight:kl_f={}-kl_z={}-0-{}-sche_{}-note={}'.format(
        opt.nEpoch, opt.batch_size, opt.decoder, opt.image_width, opt.image_width, opt.rnn_size, opt.g_dim, opt.f_dim,
        opt.z_dim, opt.lr,
        opt.weight_f, opt.weight_z, opt.loss_recon, opt.sche, opt.note)

    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    log = os.path.join(opt.log_dir, 'log.txt')

    summary_dir = os.path.join('./summary/', opt.dataset, name)
    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    print_log("Random Seed: {}".format(opt.seed), log)
    os.makedirs(summary_dir, exist_ok=True)

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)

    # control the sequence sample
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    print_log('Running parameters:')
    # print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), log)

    # ---------------- optimizers ----------------
    opt.optimizer = optim.Adam
    dbse = DBSE(opt)
    trainable_params = sum(p.numel() for p in dbse.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    dbse.apply(utils.init_weights)
    optimizer = opt.optimizer(dbse.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.sche == "cosine":
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.nEpoch+1)//2, eta_min=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4,
                                                                         T_0=(opt.nEpoch + 1) // 2, T_mult=1)
    elif opt.sche == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.nEpoch // 2, gamma=0.5)
    elif opt.sche == "const":
        scheduler = None
    else:
        raise ValueError('unknown scheduler')

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.device_count() > 1:
        print_log("Let's use {} GPUs!".format(torch.cuda.device_count()), log)
        opt.device = 'cuda'
        dbse = nn.DataParallel(dbse)

    dbse = dbse.cuda()
    print_log(dbse, log)

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

    epoch_loss = Loss()

    # --------- set RNG for permutations -------------------------
    opt.rng = np.random.default_rng(1234)

    # --------- training loop ------------------------------------
    cur_step = 0
    for epoch in range(opt.nEpoch):
        if epoch and scheduler is not None:
            scheduler.step()

        dbse.train()
        epoch_loss.reset()

        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(maxval=len(train_loader)).start()
        for i, data in enumerate(train_loader):

            progress.update(i + 1)
            x, label_A, label_D = reorder(data['video']), data['A_label'], data['D_label']
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            recon_seq, recon_frame, kld_f, kld_z = train(x, dbse, optimizer, opt)

            lr = optimizer.param_groups[0]['lr']
            if opt.neptune:
                run['train/lr'].log(lr)
                run['train/mse_seq'].log(recon_seq.item())
                run['train/mse_frame'].log(recon_frame.item())
                run['train/kld_f'].log(kld_f.item())
                run['train/kld_z'].log(kld_z.item())
                cur_step += 1

            epoch_loss.update(recon_seq, recon_frame, kld_f, kld_z)

        progress.finish()
        utils.clear_progressbar()
        avg_loss = epoch_loss.avg()
        print_log('[%02d] recon_seq: %.2f | recon_frame: %.2f | kld_f: %.2f | kld_z: %.2f | lr: %.5f'
                  % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], lr), log)

        if epoch % opt.evl_interval == 0 or epoch == opt.nEpoch - 1:
            dbse.eval()
            # save the model
            net2save = dbse.module if torch.cuda.device_count() > 1 else dbse
            torch.save({
                'model': net2save.state_dict(),
                'optimizer': optimizer.state_dict()},
                '%s/model%d.pth' % (opt.log_dir, opt.nEpoch))

        if epoch == opt.nEpoch - 1 or epoch % opt.evl_interval == 0:
            val_mse_seq = val_mse_frame = val_kld_f = val_kld_z = 0.
            for i, data in enumerate(test_loader):
                x, label_A, label_D = reorder(data['video']), data['A_label'], data['D_label']
                x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

                with torch.no_grad():
                    recon_seq, recon_frame, kld_f, kld_z = train(x, dbse, optimizer, opt, mode="val")

                val_mse_seq += recon_seq
                val_mse_frame += recon_frame
                val_kld_f += kld_f
                val_kld_z += kld_z

            n_batch = len(test_loader)

            if opt.neptune:
                run['test/mse_seq'].log(val_mse_seq.item() / n_batch)
                run['test/mse_frame'].log(val_mse_frame.item() / n_batch)
                run['test/kld_f'].log(val_kld_f.item() / n_batch)
                run['test/kld_z'].log(val_kld_z.item() / n_batch)

        if epoch == opt.nEpoch - 1 or (epoch % opt.evl_interval == 0 and epoch):
            opt.type_gt = 'action'
            a_action, _, _, _, _ = check_cls(opt, dbse, classifier, test_loader, run)
            opt.type_gt = 'aaction'
            a_action, _, _, _, _ = check_cls(opt, dbse, classifier, test_loader, run)

    if opt.neptune:
        run.stop()
    print("Training is complete")

    return a_action


class Loss(object):
    def __init__(self):
        self.reset()

    def update(self, recon_seq, recon_frame, kld_f, kld_z):
        self.recon_seq.append(recon_seq)
        self.recon_frame.append(recon_frame)
        self.kld_f.append(kld_f)
        self.kld_z.append(kld_z)

    def reset(self):
        self.recon_seq = []
        self.recon_frame = []
        self.kld_f = []
        self.kld_z = []

    def avg(self):
        return [np.asarray(i).mean() for i in
                [self.recon_seq, self.recon_frame, self.kld_f, self.kld_z]]


def get_experiment_space():
    space = {  # Architecture parameters
        'g_dim': hp.choice('g_dim', [128, 64, 256]),
        'rnn_size': hp.choice('rnn_size', [256, 128, 64]),
        'l_dim': hp.choice('l_dim', [32, 64]),
        'weight_rec_seq': hp.choice('weight_rec_seq', [5, 10, 50, 85, 100, 250]),
        'weight_rec_frame': hp.choice('weight_rec_frame', [.5, 1, 5, 10, 50, 100])}

    return space


def objective(hyperparameters):
    return main(hyperparameters)


if __name__ == '__main__':
    main(opt)
    # trials = Trials()
    # best = fmin(fn=objective, space=get_experiment_space(), algo=tpe.suggest, max_evals=300)

    # for g_dim in [128, 64, 256]:
    #     opt.g_dim = g_dim
    #     for rnn_size in [256, 128, 64]:
    #         opt.rnn_size = rnn_size
    #         for l_dim in [32, 64, 128]:
    #             opt.f_dim = l_dim
    #             opt.z_dim = l_dim
    #             for weight_rec_seq in [5, 10, 50, 100, 250]:
    #                 opt.weight_rec_seq = weight_rec_seq
    #                 for weight_rec_frame in [.5, 1, 5, 10, 50, 100]:
    #                     opt.weight_rec_frame = weight_rec_frame
    #                     main(opt)
