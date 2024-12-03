import argparse
import json
import os
import sys
import random
import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from time_series.data_loaders.data_loaders import physionet_data_loader
from time_series.model.dbse import Encoder, Decoder, DBSE

print_losses = lambda type, loss, nll, nll_stat, kld_f, kld_z: \
    "{} loss = {:.3f} \t NLL = {:.3f} \t NLL_STAT = {:.3f} \t KL_f = {:.3f} \t KL_z = {:.3f}".\
        format(type, loss, nll, nll_stat, kld_f, kld_z)

file_name_dbse = 'dbse'
file_name_predictor = 'predictor'

class Predictor(nn.Module):
    """Simple classifier layer to classify the subgroup of data

    Args:
        fc_sizes: Hidden size of the predictor MLP. default: [32, 8]
    """

    def __init__(self, rnn_size, fc_sizes, f_dim, z_dim):
        super(Predictor, self).__init__()

        self.rnn_size = rnn_size
        self.fc_sizes = fc_sizes
        self.f_dim = f_dim
        self.z_dim = z_dim

        self.rnn = nn.LSTM(input_size=z_dim, hidden_size=rnn_size, batch_first=True)

        layers = []
        in_features = rnn_size + f_dim  # 44
        for out_features in fc_sizes:
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        self.fc = nn.Sequential(*layers)

        self.dropout = nn.Dropout(p=0.5)

        # Final probability layer
        self.prob = nn.Linear(out_features, 1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, local_encs, global_encs, x_lens):
        x_lens = x_lens.to(dtype=torch.int32)
        h, _ = self.rnn(local_encs) # [10, 20, 32]
        h = torch.stack([h[i, x_lens[i]-1,:] for i in range(len(h))])

        if not global_encs is None:
            h = torch.cat([h, global_encs], dim=-1)

        logits = self.dropout(self.fc(h))
        probs = self.sigmoid(self.prob(logits))
        return probs.squeeze(-1)

def compute_loss(x, model, x_lens, args, m_mask=None, return_parts=False):
    assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
    x = x.repeat(model.M, 1, 1)  # shape=(M*BS, TL, D)

    if m_mask is not None:
        m_mask = m_mask.repeat(model.M, 1, 1)

    f_post_mean, f_post_logvar, f_post, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x, px_hat, recon_x_frame, px_hat_stat = model(x, x_lens, m_mask)

    # Compute the negative log likelihood
    nll = -px_hat.log_prob(x)
    nll_stat = -px_hat_stat.log_prob(x[:, 0, :][:, None])

    # Apply mask if provided
    if m_mask is not None:
        nll = torch.where(m_mask == 1, torch.zeros_like(nll), nll)

    # KL divergence of f and z_t
    f_post_mean = f_post_mean.view((-1, f_post_mean.shape[-1]))
    f_post_logvar = f_post_logvar.view((-1, f_post_logvar.shape[-1]))
    kld_f = -0.5 * torch.sum(1 + f_post_logvar - torch.pow(f_post_mean, 2) - torch.exp(f_post_logvar))

    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    batch_size = x.shape[0]
    kld_f, kld_z = kld_f / batch_size, kld_z / batch_size

    nll = torch.mean(nll, dim=[1, 2])
    nll_stat = torch.mean(nll_stat, dim=[1, 2])

    elbo = - nll * args.weight_rec - kld_f * args.weight_f - kld_z * args.weight_z - nll_stat * args.weight_rec_stat
    elbo = elbo.mean()
    if return_parts:
        return -elbo, nll.mean(), nll_stat.mean(), kld_z, kld_z
    return -elbo


def run_epoch(args, model, data_loader, optimizer, train=True, test=False):
    model.dataset_size = len(data_loader)
    LOSSES = []
    for i, data in enumerate(data_loader):
        x_seq, mask_seq, x_lens = data[0].cuda(), data[1].cuda(), data[2].cuda()

        if train:
            model.train()
            model.zero_grad()
            losses = compute_loss(x_seq, model, x_lens, args, m_mask=mask_seq, return_parts=True)
            if mask_seq is not None:
                losses[0].backward()
            else:
                losses.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                losses = compute_loss(x_seq, model, x_lens, args, m_mask=mask_seq, return_parts=True)

        losses = np.asarray([loss.detach().cpu().numpy() for loss in losses])
        LOSSES.append(losses)
    LOSSES = np.stack(LOSSES)
    return np.mean(LOSSES, axis=0)


def run_epoch_predictor(predictor_model, data_loader, rep_model, optimizer=None, train=True, test=False, trainable_vars=None):
    """Training epoch for training the classifier"""
    bce_loss = nn.BCELoss()
    epoch_loss, epoch_acc, epoch_auroc = [], [], []
    all_labels, all_predictions = [], []

    for i, data in enumerate(data_loader):
        x_seq, mask_seq, x_lens = data[0].cuda(), data[1].cuda(), data[2].cuda()

        labels = data[4][:, -1]

        with torch.no_grad():
            _, _, f_post, _, _, z_post, _, _, _, _, _, _, _ = rep_model(x_seq, mask_seq)
        f_post = f_post.detach()
        z_post = z_post.detach()

        lens = x_lens // rep_model.window_size

        if train:
            predictor_model.train()
            predictor_model.zero_grad()
            predictions = predictor_model(z_post, f_post, lens)
            labels = labels.to(predictions.device)
            loss = bce_loss(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor_model.parameters(), max_norm=1)
            optimizer.step()
        else:
            predictor_model.eval()
            with torch.no_grad():
                predictions = predictor_model(z_post, f_post, lens)
                labels = labels.to(predictions.device)
                loss = bce_loss(predictions, labels)
            labels = labels.to(predictions.device)
        epoch_loss.append(loss.detach().cpu().numpy().mean())
        all_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    epoch_acc = average_precision_score(all_labels, all_predictions)
    epoch_auroc = (roc_auc_score(all_labels, all_predictions))
    return np.mean(epoch_loss), epoch_acc, epoch_auroc


def train_model(args, model, train_loader, validation_loader, file_name, lr=1e-4, n_epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Assuming your run_epoch function returns a tuple of losses
    for epoch in range(n_epochs + 1):
        ep_loss, ep_nll, ep_nll_stat, ep_kld_f, ep_kld_z = run_epoch(args, model, train_loader, optimizer, train=True, test=False)

        # print losses during training
        print('=' * 30)
        print(f'Epoch {epoch}, (Learning rate: {lr:.5f})')
        print(print_losses('Training', ep_loss, ep_nll, ep_nll_stat, ep_kld_f, ep_kld_z))

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            ep_loss, ep_nll, ep_nll_stat, ep_kld_f, ep_kld_z = run_epoch(args, model, validation_loader, optimizer, train=False, test=False)
        print(print_losses('Validation', ep_loss, ep_nll, ep_nll_stat, ep_kld_f, ep_kld_z))
        # save model
        torch.save(model.state_dict(), f'{args.ckpt}{file_name}')

def train_rep_model(args, n_epochs, trainset, validset, testset):
    file_name = '%s_f%f_z%f_lr%f_recseq%f_reqstat%f_%s' % (file_name_dbse, args.weight_f, args.weight_z, args.lr, args.weight_rec, args.weight_rec_stat, args.data)

    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]

    if args.train:
        print(f'Training DBSE model on {args.data} saving to file {file_name}')
        print(args)

    # Create the representation learning models
    encoder = Encoder(input_size=10, hidden_sizes=configs["dbse_encoder_size"])
    decoder = Decoder(args,
                      output_size=configs["feature_size"],
                      output_length=configs["window_size"],
                      hidden_sizes=configs["dbse_decoder_size"])
    rep_model = DBSE(encoder=encoder, decoder=decoder, configs=configs, args=args)

    rep_model.cuda()

    # Train the DBSE baseline
    if args.train:
        if not os.path.exists(f'{args.ckpt}'):
            os.mkdir(f'{args.ckpt}')

        train_model(args, rep_model, trainset, validset, lr=args.lr, n_epochs=n_epochs, file_name=file_name)

    # Load model weights
    checkpoint_path = f'{args.ckpt}{file_name}'
    rep_model.load_state_dict(torch.load(checkpoint_path))
    rep_model.eval()  # Set the model to evaluation mode

    # Report test performance
    test_loss, test_nll, test_nll_stat, test_kld_f, test_kld_z = run_epoch(args, rep_model, testset, optimizer=None, train=False, test=True)
    print(f'\nDBSE performance on {args.data} data')
    print('{}'.format(print_losses('Test', test_loss, test_nll, test_nll_stat, test_kld_f, test_kld_z)))

    return rep_model


def train_predictor_and_test_mortality(args, trainset, validset, testset, rep_model):
    file_name = '%s_f%f_z%f_lr%f_recseq%f_reqstat%f_%s' % (file_name_predictor, args.weight_f, args.weight_z, args.lr, args.weight_rec, args.weight_rec_stat, args.data)

    rep_model.eval()  # Set the model to evaluation mode

    ##### ----- finished training our model, now train downstream task ----- #####
    predictor_model = Predictor(32, [16], args.f_dim, args.z_dim)
    predictor_model.cuda()
    n_epochs = 40
    lr = args.lr

    optimizer = torch.optim.Adam(predictor_model.parameters(), lr=lr)

    losses_train, acc_train, auroc_train = [], [], []
    losses_val, acc_val, auroc_val = [], [], []
    for epoch in range(n_epochs + 1):
        epoch_loss_train, epoch_acc_train, epoch_auroc_train = run_epoch_predictor(predictor_model, trainset, rep_model, optimizer=optimizer, train=True, test=False, trainable_vars=None)
        if epoch % 1 == 0:
            print('=' * 30)
            print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
            losses_train.append(epoch_loss_train)
            acc_train.append(epoch_acc_train)
            auroc_train.append(epoch_auroc_train)
            print("Training loss = %.3f \t Accuracy = %.3f \t AUROC = %.3f" % (
                epoch_loss_train, epoch_acc_train, epoch_auroc_train))
            epoch_loss_val, epoch_acc_val, epoch_auroc_val = run_epoch_predictor(predictor_model, validset, rep_model, train=False)
            losses_val.append(epoch_loss_val)
            acc_val.append(epoch_acc_val)
            auroc_val.append(epoch_auroc_val)
            print("Validation loss = %.3f \t Accuracy = %.3f \t AUROC = %.3f" % (
                epoch_auroc_val, epoch_acc_val, epoch_auroc_val))


    test_loss, test_acc, test_auroc = run_epoch_predictor(predictor_model, testset, rep_model, train=False)
    print("\n Test performance \t loss = %.3f \t AUPRC = %.3f \t AUROC = %.3f" % (
                        test_loss, test_acc, test_auroc))

    torch.save(predictor_model.state_dict(), f'{args.ckpt}{file_name}')
    return test_loss, test_acc, test_auroc


def define_seed(seed):
    # Control the sequence sample
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    # All default argument are set by default to the best results that we reported on our paper
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--ckpt', type=str, default='./trained_models/')
    parser.add_argument('--data', type=str, default='physionet', help="dataset to use")
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    parser.add_argument('--rep_size', type=int, default=8, help="Size of the representation vectors")
    parser.add_argument('--f_dim', type=int, default=8, help="Size of static vector")
    parser.add_argument('--z_dim', type=int, default=8, help="Size of dynamic vector")
    parser.add_argument('--rnn_size', default=64, type=int, help='dimensionality of hidden layer')
    parser.add_argument('--g_dim', default=32, type=int, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--frame_ind', default=0, type=float, help='frame index for to learn the static information from')

    parser.add_argument('--weight_rec', type=float, default=90., help='weighting on general reconstruction')
    parser.add_argument('--weight_rec_stat', type=float, default=9., help='weighting on general reconstruction')
    parser.add_argument('--weight_f', type=float, default=0.2, help='weighting on KL to prior, content vector')
    parser.add_argument('--weight_z', type=float, default=0.01, help='weighting on KL to prior, motion vector')

    parser.add_argument('--note', default='sample', type=str, help='appx note')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')

    parser.add_argument('--data_dir', type=str, default=None, help='path to the directory of the data')
    parser.add_argument('--physionet_dataset_path', type=str, default=None, help='path to the physionet data .csv')
    parser.add_argument('--physionet_static_dataset_path', type=str, default=None, help='path to the physionet static data .csv')

    args = parser.parse_args()
    define_seed(args)
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]

    n_epochs = 250
    trainset, validset, testset, _ = physionet_data_loader(args.data_dir, args.physionet_dataset_path, args.physionet_static_dataset_path, window_size=configs["window_size"], frame_ind=args.frame_ind, normalize="mean_zero")


    rep_model = None
    cv_loss, cv_acc, cv_auroc = [], [], []

    for cv in range(3):
        if cv == 0:
            rep_model = train_rep_model(args, n_epochs, trainset, validset, testset)
        test_loss, test_acc, test_auroc = train_predictor_and_test_mortality(args, trainset, validset, testset, rep_model)
        cv_loss.append(test_loss)
        cv_acc.append(test_acc)
        cv_auroc.append(test_auroc)

    print("loss = %.3f $\pm$ %.3f \t AUPRC = %.3f $\pm$ %.3f \t AUROC = %.3f $\pm$ %.3f" % (np.mean(cv_loss), np.std(cv_loss), np.mean(cv_acc), np.std(cv_acc), np.mean(cv_auroc),np.std(cv_auroc)))

