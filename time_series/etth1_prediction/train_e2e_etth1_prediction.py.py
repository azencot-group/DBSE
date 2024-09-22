import argparse
import json
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from ..data_loaders.data_loaders import etth_data_loader
from ..model.dbse import Encoder, Decoder, DBSE

print_losses = lambda type, loss, nll, nll_stat, kld_f, kld_z: \
    "{} loss = {:.3f} \t SEQ_LOSS = {:.3f} \t FRAME_LOSS = {:.3f} \t KL_f = {:.3f} \t KL_z = {:.3f}". \
        format(type, loss, nll, nll_stat, kld_f, kld_z)

file_name_dbse = 'dbse'
file_name_predictor = 'predictor'


class Predictor(nn.Module):
    """Simple classifier layer to classify the subgroup of data

    Args:
        fc_sizes: Hidden size of the predictor MLP. default: [32, 8]
    """

    def __init__(self, fc_sizes=[32, 8], args=None):
        super(Predictor, self).__init__()

        # Define the fully connected (fc) layers
        layers = []
        in_features = args.f_dim + args.z_dim
        for out_features in fc_sizes:
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.LeakyReLU())
            in_features = out_features
        self.fc = nn.Sequential(*layers)

        # Final probability layer
        self.prob = nn.Linear(in_features=fc_sizes[-1], out_features=1)

        self.relu = nn.LeakyReLU()

        # Batch Normalization Layer
        self.batch_norm = nn.BatchNorm1d(args.f_dim + args.z_dim)  # Apply normalization across the 16 sequences

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, local_encs, global_encs, x_lens):
        # local_encs: [10,12], global_encs: [10, 28, 4], x_lens: [10]
        if global_encs is not None:
            global_encs_expanded = global_encs.unsqueeze(1).repeat(1, local_encs.shape[1], 1)  # [10, 28, 12]
            h = torch.cat([local_encs, global_encs_expanded], dim=2)  # [10, 28, 16]

            # h will have shape: [5, 10, 16]
            h = self.batch_norm((h.transpose(1, 2))).transpose(1, 2)
        else:
            h = local_encs

        logits = self.fc(h)  # [10, 28, 8]
        probs = self.relu(self.prob(logits))  # [10, 28, 1]
        return probs.squeeze(-1)  # [5, 10]


def compute_loss(x, model, x_lens, args, m_mask=None, return_parts=False):
    assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
    x = x.repeat(model.M, 1, 1)  # shape=(M*BS, TL, D)

    if m_mask is not None:
        m_mask = m_mask.repeat(model.M, 1, 1)

    f_post_mean, f_post_logvar, f_post, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x, px_hat, recon_x_frame, px_hat_stat = model(
        x, x_lens, m_mask)

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
            losses[0].backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                losses = compute_loss(x_seq, model, x_lens, args, m_mask=mask_seq, return_parts=True)

        losses = np.asarray([loss.detach().cpu().numpy() for loss in losses])

        LOSSES.append(losses)
    LOSSES = np.stack(LOSSES)
    return np.mean(LOSSES, axis=0)


def run_epoch_predictor(predictor_model, data_loader, rep_model, optimizer=None, label_blocks=None, train=True, test=False, trainable_vars=None):
    """Training epoch for training the classifier"""
    mae = nn.L1Loss()
    epoch_loss = []

    b_start = 0
    for i, data in enumerate(data_loader):
        x_seq, mask_seq, x_lens = data[0].cuda(), data[1].cuda(), data[2].cuda()

        mask_seq = None

        labels = label_blocks[b_start:b_start + len(x_seq)]
        b_start += len(x_seq)
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels)

        with torch.no_grad():
            _, _, f_post, _, _, z_post, _, _, _, _, _, _, _ = rep_model(x_seq, mask_seq)
        f_post = f_post.detach()
        z_post = z_post.detach()

        lens = x_lens // rep_model.window_size  # lens.shape = [10]

        if train:
            predictor_model.train()
            predictor_model.zero_grad()
            predictions = predictor_model(z_post, f_post, lens)
            labels = labels.to(predictions.device)

            loss = mae(labels, predictions)
            loss.backward()
            optimizer.step()
        else:
            predictor_model.eval()
            with torch.no_grad():
                predictions = predictor_model(z_post, f_post, lens)
            labels = labels.to(predictions.device)  # labels.shape = [10, 28]
        epoch_loss.append(mae(labels, predictions).detach().cpu().numpy().mean())
    return np.mean(epoch_loss)


def train_mo_mi(args, model, train_loader, validation_loader, file_name, lr=1e-4, n_epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Assuming your run_epoch function returns a tuple of losses
    for epoch in range(n_epochs + 1):
        ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z = run_epoch(args, model, train_loader, optimizer,
                                                                            train=True, test=False)

        # print losses during training
        print('=' * 30)
        print(f'Epoch {epoch}, (Learning rate: {lr:.5f})')
        print(print_losses('Training', ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z))

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z = run_epoch(args, model, validation_loader,
                                                                                optimizer, train=False, test=False)
        print(print_losses('Validation', ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z))

        # save model
        torch.save(model.state_dict(), f'{args.ckpt}{file_name}')


def block_labels(data_loader, rep_model):
    window_size = rep_model.window_size

    # List to store the selected labels from each batch
    selected_labels_list = []

    # Loop through each batch in the data loader
    for batch in data_loader:
        # Get the selected labels from the current batch
        selected_labels = batch[3][:, :, 1]
        selected_labels_list.append(selected_labels)

    # Concatenate all the selected labels along the first dimension
    all_labels_blocks = torch.cat(selected_labels_list, dim=0)

    # Split the tensor along the 1-axis (time dimension)
    all_labels_blocks = torch.split(all_labels_blocks, window_size, dim=1)

    # Calculate the average of each block along the 1-axis (time dimension)
    # and stack them along the last dimension
    all_labels_blocks = torch.stack([torch.mean(block, dim=1) for block in all_labels_blocks], dim=-1)

    return all_labels_blocks


def train_rep_model(args, n_epochs, trainset, validset, testset):
    file_name = '%s_f%f_z%f_lr%f_recseq%f_reqstat%f_%s' % (
        file_name_dbse, args.weight_f, args.weight_z, args.lr, args.weight_rec, args.weight_rec_stat, args.data)

    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]

    define_seed(args)
    if args.train:
        print(f'Training DBSE model on {args.data} saving to file {file_name}')
        print(args)

    # Create the representation learning models
    encoder = Encoder(input_size=configs["feature_size"], hidden_sizes=configs["dbse_encoder_size"])
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

        train_mo_mi(args, rep_model, trainset, validset, lr=args.lr, n_epochs=n_epochs, file_name=file_name)

    # Load model weights
    checkpoint_path = f'{args.ckpt}{file_name}'
    rep_model.load_state_dict(torch.load(checkpoint_path))
    rep_model.eval()  # Set the model to evaluation mode

    # Report test performance
    test_loss, test_nll, test_nll_stat, test_kld_f, test_kld_z = run_epoch(args, rep_model, testset, optimizer=None,
                                                                           train=False, test=True)
    print(f'\nDBSE performance on {args.data} data')
    print('{}'.format(print_losses('Test', test_loss, test_nll, test_nll_stat, test_kld_f, test_kld_z)))

    return rep_model


def train_predictor_and_test_avg_oil_temp(args, trainset, validset, testset, rep_model):
    file_name = '%s_f%f_z%f_lr%f_recseq%f_reqstat%f_%s' % (
        file_name_predictor, args.weight_f, args.weight_z, args.lr, args.weight_rec, args.weight_rec_stat, args.data)

    rep_model.eval()  # Set the model to evaluation mode

    ##### ----- finished training our model, now train downstream task ----- #####
    test_loss = []
    label_blocks_train = block_labels(trainset, rep_model)
    label_blocks_train.cuda()
    label_blocks_valid = block_labels(validset, rep_model)
    label_blocks_valid.cuda()
    label_blocks_test = block_labels(testset, rep_model)
    label_blocks_test.cuda()

    for cv in range(3):
        predictor_model = Predictor([32, 8], args)
        predictor_model.cuda()
        n_epochs = 300
        lr = 0.001

        optimizer = torch.optim.Adam(predictor_model.parameters(), lr=lr)
        losses_train = []
        losses_val = []
        for epoch in range(n_epochs + 1):
            epoch_loss_train = run_epoch_predictor(predictor_model, trainset, rep_model, optimizer=optimizer,
                                                   label_blocks=label_blocks_train,
                                                   train=True, test=False, trainable_vars=None)
            if epoch and epoch % 5 == 0:
                print('=' * 30)
                print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
                losses_train.append(epoch_loss_train)

                epoch_loss_val = run_epoch_predictor(predictor_model, validset, rep_model,
                                                     label_blocks=label_blocks_valid, train=False, test=False)
                losses_val.append(epoch_loss_val)
                te_loss = run_epoch_predictor(predictor_model, testset, rep_model, label_blocks=label_blocks_test,
                                              train=False, test=True)

                print("Training loss = %.3f" % (epoch_loss_train))
                print("Validation loss = %.3f" % (epoch_loss_val))
                print('Test loss =  %.3f' % (te_loss))

        test_loss.append(
            run_epoch_predictor(predictor_model, testset, rep_model, label_blocks=label_blocks_test, train=False))

    return test_loss


def define_seed(seed):
    # Control the sequence sample
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # All default argument are set by default to the best results that we reported on our paper
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--ckpt', type=str, default='./trained_models/')
    parser.add_argument('--data', type=str, default='etth', help="dataset to use")
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    parser.add_argument('--rep_size', type=int, default=8, help="Size of the representation vectors")
    parser.add_argument('--f_dim', type=int, default=8, help="Size of static vector")
    parser.add_argument('--z_dim', type=int, default=8, help="Size of dynamic vector")
    parser.add_argument('--rnn_size', default=64, type=int, help='dimensionality of hidden layer')
    parser.add_argument('--g_dim', default=32, type=int, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--frame_ind', default=14, type=float, help='frame index for to learn the first frame')

    parser.add_argument('--weight_rec', type=float, default=85., help='weighting on general reconstruction')
    parser.add_argument('--weight_rec_stat', type=float, default=11., help='weighting on general reconstruction')
    parser.add_argument('--weight_f', type=float, default=0.01, help='weighting on KL to prior, content vector')
    parser.add_argument('--weight_z', type=float, default=0.001, help='weighting on KL to prior, motion vector')

    parser.add_argument('--note', default='sample', type=str, help='appx note')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')

    args = parser.parse_args()

    define_seed(args)

    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]

    n_epochs = 300
    trainset, validset, testset, _ = etth_data_loader(window_size=configs["window_size"], frame_ind=args.frame_ind, normalize="mean_zero")

    cv_loss, cv_acc, cv_auroc = [], [], []


    rep_model = None
    rep_model = train_rep_model(args, n_epochs, trainset, validset, testset)
    test_loss, test_acc, test_auroc = train_predictor_and_test_avg_oil_temp(args, trainset, validset, testset, rep_model)

    print("\n\n Final performance \t loss = %.3f +- %.3f" % (np.mean(test_loss), np.std(test_loss)))
