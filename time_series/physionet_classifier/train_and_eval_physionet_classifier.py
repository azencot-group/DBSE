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

from time_series.data_loaders.data_loaders import physionet_data_loader
from time_series.model.dbse import Encoder, Decoder, DBSE


print_losses = lambda type, loss, nll, nll_stat, kld_f, kld_z: \
    "{} loss = {:.3f} \t SEQ_LOSS = {:.3f} \t FRAME_LOSS = {:.3f} \t KL_f = {:.3f} \t KL_z = {:.3f}".\
        format(type, loss, nll, nll_stat, kld_f, kld_z)

file_name_dbse = 'dbse'
file_name_classifier = 'classifier'

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

def train_mo_mi(args, model, train_loader, validation_loader, file_name, lr=1e-4, n_epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Assuming your run_epoch function returns a tuple of losses
    for epoch in range(n_epochs + 1):
        ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z = run_epoch(args, model, train_loader, optimizer, train=True, test=False)

        # print losses during training
        print('=' * 30)
        print(f'Epoch {epoch}, (Learning rate: {lr:.5f})')
        print(print_losses('Training', ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z))
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z = run_epoch(args, model, validation_loader, optimizer, train=False, test=False)
        print(print_losses('Validation', ep_loss, ep_seq_loss, ep_frame_loss, ep_kld_f, ep_kld_z))

        # save model
        torch.save(model.state_dict(), f'{args.ckpt}{file_name}')

def train_rep_model(args, n_epochs, trainset, validset, testset):
    file_name = '%s_f%f_z%f_lr%f_recseq%f_reqstat%f_%s' % (file_name_dbse, args.weight_f, args.weight_z, args.lr, args.weight_rec, args.weight_rec_stat, args.data)

    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]


    define_seed(args)
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

    # Train the DBSE
    if args.train:
        if not os.path.exists(f'{args.ckpt}'):
            os.mkdir(f'{args.ckpt}')

        train_mo_mi(args, rep_model, trainset, validset, lr=args.lr, n_epochs=n_epochs, file_name=file_name)

    # Load model weights
    checkpoint_path = f'{args.ckpt}{file_name}'
    rep_model.load_state_dict(torch.load(checkpoint_path))
    rep_model.eval()  # Set the model to evaluation mode

    # Report test performance
    test_loss, test_nll, test_nll_stat, test_kld_f, test_kld_z = run_epoch(args, rep_model, testset, optimizer=None, train=False, test=True)
    print(f'\nDBSE performance on {args.data} data')
    print('{}'.format(print_losses('Test', test_loss, test_nll, test_nll_stat, test_kld_f, test_kld_z)))

    return rep_model


def train_classifier(trainset, validset, classifier_model, rep_model, n_epochs, lr, data, args, file_name):
    """Train a classifier to classify the subgroup of time series"""
    losses_train, losses_val, acc_train, acc_val = [], [], [], []
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    validation_loss = float('inf')
    for epoch in range(n_epochs+1):
        train_loss, train_acc = run_classifier_epoch(classifier_model, rep_model, trainset, optimizer= optimizer, data=data, train=True)
        valid_loss, valid_acc = run_classifier_epoch(classifier_model, rep_model, validset, optimizer=optimizer, data=data, train=False)
        losses_train.append(train_loss)
        acc_train.append(train_acc)
        losses_val.append(valid_loss)
        acc_val.append(valid_acc)

        if valid_loss < validation_loss:
            validation_loss = valid_loss
            torch.save(classifier_model.state_dict(), f'{args.ckpt}{file_name}')

        if epoch % 5 == 0:
            print('=' * 30)
            print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
            print("Training loss = %.3f \t Training accuracy = %.3f" % (train_loss, train_acc))
            print("Validation loss = %.3f \t Validation accuracy = %.3f" % (valid_loss, valid_acc))



def run_classifier_epoch(classifier_model, rep_model, dataset, data, optimizer=None, train=False, repeat=5):
    """Training epoch of a classifier"""
    ce_loss = nn.CrossEntropyLoss()
    epoch_loss, epoch_acc= [], []
    for _ in range(repeat):
        for i, data_a in enumerate(dataset):
            x_seq, mask_seq, x_lens = data_a[0].cuda(), data_a[1].cuda(), data_a[2]
            mask_seq = mask_seq.to(torch.float32)
            if args.category=='global':
                rnd_t = np.random.randint(0, ((x_seq.shape[1] // rep_model.window_size if x_lens is None else min(x_lens)) // rep_model.window_size) - 1)

                if data=='air_quality':
                    labels = torch.tensor([int(m.item()) - 1 for m in data_a[4][:, 1]], dtype=torch.int64)

                elif data=='physionet':
                    labels = data_a[4][:, 3] - 1
                    labels = labels.to(torch.int64)

            elif args.category=='local':
                rnd_t = np.random.randint(0, ((x_seq.shape[1] if x_lens is None else min(x_lens))//rep_model.window_size)-1)

            with torch.no_grad():
                _, _, f_post, _, _, z_post, _, _, _, _, _, _, _ = rep_model(x_seq, mask_seq)
            f_post = f_post.detach()
            z_post = z_post.detach()

            if train:
                classifier_model.train()
                classifier_model.zero_grad()
                predictions = classifier_model(f_post, z_post, args)
                labels = labels.to(predictions.device)
                loss = ce_loss(predictions, labels)
                loss.backward()
                optimizer.step()
            else:
                classifier_model.eval()
                with torch.no_grad():
                    predictions = classifier_model(f_post, z_post, args)
                labels = labels.to(predictions.device)
                loss = ce_loss(predictions, labels)

            accuracy = (labels == predictions.argmax(dim=-1)).float()
            accuracy = accuracy.cpu().numpy()
            accuracy = np.mean(accuracy)

            epoch_loss.append(loss.detach().cpu().item())
            epoch_acc.append(accuracy)
    return np.mean(epoch_loss), np.mean(epoch_acc)

def classification_exp(representation_classifier, rep_model, args, data, datasets, file_name):
    """Run the classification experiment"""
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]

    print('************************* TRAINING %s CLASSIFIER *************************' % args.category)
    train_classifier(trainset, validset, classifier_model=representation_classifier, rep_model=rep_model, n_epochs=40, lr=args.lr, data=data, args=args, file_name=file_name)

    # Load classifier model weights
    checkpoint_path = f'{args.ckpt}{file_name}'
    representation_classifier.load_state_dict(torch.load(checkpoint_path))

    test_loss, test_acc = run_classifier_epoch(representation_classifier, rep_model, testset, data=data, train=False)
    print('Testset ==========> Accuracy = %.3f\n' % test_acc)
    return test_acc, test_loss

def train_and_test_classifier(args, trainset, validset, testset, rep_model):
    file_name = '%s_f%f_z%f_lr%f_recseq%f_reqstat%f_%s' % (file_name_classifier, args.weight_f, args.weight_z, args.lr, args.weight_rec, args.weight_rec_stat, args.data)

    rep_model.eval()  # Set the model to evaluation mode

    test_accuracies, test_losses = [], []
    if args.category == 'global':
        classifier_model = LinearClassifier(args.f_dim, n_classes=args.n_classes)
    else:
        classifier_model = LinearClassifier(args.z_dim, n_classes=args.n_classes)

    classifier_model.cuda()

    for cv in range(2):
        test_acc, test_loss = classification_exp(classifier_model, rep_model, args, data=args.data, datasets=(trainset, validset, testset), file_name=file_name)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

    return test_accuracies, test_losses


def define_seed(args):
    # Control the sequence sample
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


class LinearClassifier(nn.Module):
    def __init__(self, fc_input_dim, n_classes, regression=False):
        super(LinearClassifier, self).__init__()
        self.fc_input_dim = fc_input_dim
        if regression:
            self.n_classes = 1
            self.classifier = nn.Sequential(
                nn.Linear(self.fc_input_dim, self.n_classes),
                nn.ReLU(),
                nn.Linear(self.n_classes, 1),
                nn.ReLU()
            )
        else:
            self.n_classes = n_classes
            self.classifier = nn.Sequential(
                nn.Linear(self.fc_input_dim, self.n_classes),
                nn.ReLU(),
                nn.Linear(self.n_classes, self.n_classes))


    def forward(self, f_post, z_post, args):
        if args.category=='global':
            probs = self.classifier(f_post)
        else:
            probs = self.classifier(z_post)

        return probs


if __name__=="__main__":
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
    parser.add_argument('--category', default='global', type=str, help='global or local')
    parser.add_argument('--n_classes', type=int, default=4, help="number of classes")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--frame_ind', default=0, type=float, help='frame index for to learn the static information from')

    parser.add_argument('--weight_rec', type=float, default=80., help='weighting on general reconstruction')
    parser.add_argument('--weight_rec_stat', type=float, default=3., help='weighting on general reconstruction')
    parser.add_argument('--weight_f', type=float, default=0.2, help='weighting on KL to prior, content vector')
    parser.add_argument('--weight_z', type=float, default=0.2, help='weighting on KL to prior, motion vector')

    parser.add_argument('--note', default='sample', type=str, help='appx note')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')

    parser.add_argument('--data_dir', type=str, default=None, help='path to the directory of the data')
    parser.add_argument('--physionet_dataset_path', type=str, default=None, help='path to the physionet data .csv')
    parser.add_argument('--physionet_static_dataset_path', type=str, default=None, help='path to the physionet static data .csv')

    args = parser.parse_args()

    define_seed(args)

    n_epochs = 400
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]
    trainset, validset, testset, _ = physionet_data_loader(args.data_dir, args.physionet_dataset_path, args.physionet_static_dataset_path, window_size=configs["window_size"], frame_ind=args.frame_ind, normalize="mean_zero")


    rep_model = train_rep_model(args, n_epochs, trainset, validset, testset)
    test_accuracies, test_losses = train_and_test_classifier(args, trainset, validset, testset, rep_model)

    print(
        '\n Overall Test Performance ==========> Accuracy = %.2f +- %.2f \t Loss = %.2f +- %.2f' % (
            100 * np.mean(test_accuracies), 100 * np.std(test_accuracies), np.mean(test_losses),
            np.std(test_losses)))


