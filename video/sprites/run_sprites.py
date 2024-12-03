import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import video.utils.video_utils as utils
import progressbar
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from sprites_hyperparameters import *

from video.utils.video_utils import reorder
from sprites_utils_train import check_cls
from video.model.dbse_utils import DbseLoss
from video.model.dbse_model import DBSE, classifier_Sprite_all

mse_loss = nn.MSELoss().cuda()
# Constants to be defined by the user
SPRITE_JUDGE_PATH = None


# --------- training funtions ------------------------------------
def train(x, model, optimizer, opt, mode="train"):
    """
    Train or evaluate the model depending on the mode.

    :param x: Input data (batch of sequences).
    :param model: The model to train/evaluate.
    :param optimizer: Optimizer for training.
    :param opt: Configuration options.
    :param mode: "train" for training, "val" for evaluation.
    :return: List of loss components (reconstruction, KL-divergence terms).
    """
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
    """
    Main training loop for the model.

    :param opt: Configuration options.
    :return: Final action classification accuracy (a_action).
    """
    a_action = 0
    run = None
    opt.rng = 1234

    # control the sequence sample
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

    # ---------------- optimizers ----------------
    opt.optimizer = optim.Adam
    dbse = DBSE(opt)
    trainable_params = sum(p.numel() for p in dbse.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    dbse.apply(utils.init_weights)
    optimizer = opt.optimizer(dbse.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.sche == "cosine":
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
        opt.device = 'cuda'
        dbse = nn.DataParallel(dbse)

    dbse = dbse.cuda()
    old_rnn_size = opt.rnn_size
    old_g_dim = opt.g_dim
    opt.rnn_size = 256
    opt.g_dim = 128
    classifier = classifier_Sprite_all(opt)
    opt.resume = SPRITE_JUDGE_PATH
    loaded_dict = torch.load(opt.resume)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()
    opt.rnn_size = old_rnn_size
    opt.g_dim = old_g_dim

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)
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
    opt.dataset_size = len(train_data)
    epoch_loss = DbseLoss()

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

            epoch_loss.update(recon_seq, recon_frame, kld_f, kld_z)

        progress.finish()
        utils.clear_progressbar()

        if epoch % opt.evl_interval == 0 or epoch == opt.nEpoch - 1:
            dbse.eval()
            net2save = dbse.module if torch.cuda.device_count() > 1 else dbse
            # torch.save({
            #     'model': net2save.state_dict(),
            #     'optimizer': optimizer.state_dict()},
            #     '%s/model%d.pth' % (opt.log_dir, opt.nEpoch))

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

        if epoch == opt.nEpoch - 1 or (epoch % opt.evl_interval == 0 and epoch):
            opt.type_gt = 'action'
            a_action, _, _, _, _ = check_cls(opt, dbse, classifier, test_loader, run)
            opt.type_gt = 'aaction'
            a_action, _, _, _, _ = check_cls(opt, dbse, classifier, test_loader, run)

    print("Training is complete")

    return a_action


if __name__ == '__main__':
    main(opt)
