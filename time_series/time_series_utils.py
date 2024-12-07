import torch

import torch.nn as nn
import random
import numpy as np


def define_seed(args):
    # Control the sequence sample
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


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
        if args.category == 'global':
            probs = self.classifier(f_post)
        else:
            probs = self.classifier(z_post)

        return probs


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
        h, _ = self.rnn(local_encs)  # [10, 20, 32]
        h = torch.stack([h[i, x_lens[i] - 1, :] for i in range(len(h))])

        if not global_encs is None:
            h = torch.cat([h, global_encs], dim=-1)

        logits = self.dropout(self.fc(h))
        probs = self.sigmoid(self.prob(logits))
        return probs.squeeze(-1)
