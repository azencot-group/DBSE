import torch
import torch.nn as nn
import numpy as np


def reparameterize(mean, logvar, random_sampling=True):
    # Re-parametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar)  # eps ~ N(0, 1) in size of logvar tensor
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean


class DBSE(nn.Module):
    def __init__(self, opt):
        super(DBSE, self).__init__()
        self.opt = opt
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame/image feature
        self.channels = opt.channels  # image channel
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = opt.frames
        # self.frame = opt.frame

        # Frame encoder and decoder
        if opt.image_width == 64:
            from dbse_utils import encoder
            if opt.decoder == 'Conv':
                from dbse_utils import decoder_conv as decoder
                from dbse_utils import decoder_conv_static as decoder_static
            elif opt.decoder == 'ConvT':
                from dbse_utils import decoder_convT as decoder
                from dbse_utils import decoder_convT_static as decoder_static
            else:
                raise ValueError('no implementation of decoder {}'.format(opt.decoder))
        else:
            raise ValueError('Not implementated for image size {}.'.format(opt.image_width))

        self.encoder = encoder(self.g_dim, self.channels)
        self.decoder = decoder(self.f_dim + self.z_dim, self.channels)

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # ----- Posterior of content and motion
        # content and motion features share one bi-lstm
        # self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.z_lstm = torch.nn.LSTMCell(self.g_dim, self.hidden_dim)

        self.f_mean = nn.Linear(self.hidden_dim, self.f_dim)
        self.f_logvar = nn.Linear(self.hidden_dim, self.f_dim)

        self.z_mean = nn.Linear(self.z_dim * 2, self.z_dim)
        self.z_logvar = nn.Linear(self.z_dim * 2, self.z_dim)

        # ----- Static features
        self.static_feature_extractor = nn.Sequential(
            nn.Linear(self.g_dim, self.hidden_dim),  # First linear layer
            nn.Tanh()
        )

        # LSTM for dynamics
        self.z_lstm = nn.LSTM(self.g_dim, self.z_dim, 2, bidirectional=True, batch_first=True)

    def encode_and_sample_post(self, x):
        conv_x = self.encoder_frame(x)  # conv_x: (batch_size, 15, 128)
        bsz, seq_len, h_dim = conv_x.shape

        # Subtract conv_x first frame from all others conv_x frames
        conv_x_static = conv_x[:, 0][:, None, :]
        conv_x_dynamic = conv_x - conv_x[:, 0][:, None, :]

        # ---- compute static features ----
        f = self.static_feature_extractor(conv_x_static)

        f_mean = self.f_mean(f)
        f_logvar = self.f_logvar(f)
        f_post = reparameterize(f_mean, f_logvar, random_sampling=True)

        # ---- compute dynamics ----
        # Rewrite z as z from the second frame
        # concatenate zeros on first frame of z
        first_frame_dynamic = reparameterize(torch.zeros((conv_x_dynamic.shape[0], conv_x_dynamic.shape[-1])).cuda(),
                                             torch.zeros((conv_x_dynamic.shape[0], conv_x_dynamic.shape[-1])).cuda(),
                                             random_sampling=True)
        z = torch.cat((first_frame_dynamic[:, None], conv_x_dynamic[:, 1:]), dim=1)
        z = self.z_lstm(z)[0]

        z_mean = self.z_mean(z)
        z_logvar = self.z_logvar(z)
        z_post = reparameterize(z_mean, z_logvar, random_sampling=True)

        # f_mean is list if triple else not
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

    # ------ sample z from learned LSTM prior base on previous postior, teacher forcing for training  ------
    def sample_motion_prior_train(self, z_post, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # Two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:, i, :]
        return z_means, z_logvars, z_out

    # ------ Sample z purely from learned LSTM prior with arbitrary frames ------
    def sample_motion_prior(self, n_sample, n_frame, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = n_sample

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(n_frame):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def forward(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_motion_prior_train(z_post, random_sampling=True)

        f_expand = f_post.expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_seq_x = self.decoder(zf)
        recon_frame_x = self.decoder(torch.cat((z_post[:, 0][:, None], f_post), dim=2)).squeeze(1)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, recon_frame_x, recon_seq_x

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    # fixed content and sample motion for classification disagreement scores
    def forward_fixed_content_for_classification(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_out = self.sample_motion_prior(x.size(0), self.frames, random_sampling=True)

        f_expand = f_mean.expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_prior = reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                 random_sampling=True)
        f_expand = f_prior.expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_post.expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    # sample content and fixed motion for classification disagreement scores
    def forward_sample_action_and_static(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        f_prior = reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                 random_sampling=True)
        f_expand = f_prior.expand(-1, self.frames, self.f_dim)
        z_mean_prior, _, _ = self.sample_motion_prior(x.size(0), self.frames, random_sampling=True)
        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return f_prior, z_mean_prior, recon_x_sample

    def forward_fixed_action_for_classification_multi(self, x):
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        f_prior = reparameterize(torch.zeros((x.shape[0] * x.shape[0], self.f_dim)).cuda(),
                                 torch.zeros((x.shape[0] * x.shape[0], self.f_dim)).cuda(),
                                 random_sampling=True)
        # f_prior = reparameterize(f_mean, torch.zeros(f_logvar.shape).cuda(), random_sampling=True)
        f_expand = f_prior.expand(-1, self.frames, self.f_dim)
        zf = torch.cat(
            (z_mean_post[None].expand(x.shape[0], -1, -1, -1).reshape(-1, self.frames, self.z_dim), f_expand), dim=2)
        # zf = torch.cat((z_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    def forward_fixed_content_for_classification_multi(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_out = \
            self.sample_motion_prior_train(z_post[None].expand(x.shape[0], -1, -1, -1).
                                           reshape(-1, self.frames, self.z_dim), random_sampling=True)

        f_expand = f_mean.expand(-1, self.frames, self.f_dim).repeat_interleave(x.shape[0], dim=0)
        zf = torch.cat((z_out, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    def forward_fixed_content_for_classification_tr(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        z_mean_prior, z_logvar_prior, z_out = self.sample_motion_prior(x.size(0), self.frames, random_sampling=True)

        f_expand = f_mean.expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification_tr(self, x):
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        f_prior = reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                 random_sampling=True)
        # f_prior = reparameterize(f_mean, torch.zeros(f_logvar.shape).cuda(), random_sampling=True)
        f_expand = f_prior.expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        # zf = torch.cat((z_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    def samples_tr(self, sz=128):
        # sz means the number of samples
        f_shape = (sz, self.f_dim)

        # sample f
        f_prior = reparameterize(torch.zeros(f_shape).cuda(), torch.zeros(f_shape).cuda(), random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        # sample z
        z_mean_prior, z_logvar_prior, z_out = self.sample_motion_prior(sz, self.frames, random_sampling=True)
        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(recon_x_sample)

        return f_mean, f_logvar, torch.mean(z_mean_post, dim=1), torch.mean(z_logvar_post, dim=1)

    def forward_exchange(self, x):
        f_mean, f_logvar, f, z_mean_post, z_logvar_post, z = self.encode_and_sample_post(x)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]

        a = f[np.arange(0, f.shape[0], 2)]
        b = f[np.arange(1, f.shape[0], 2)]
        f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, None, None, z, None, None, recon_x

    def forward_for_swap_dynamic_and_static(self, x1, x2):
        f1_mean, _, _, z1_mean, _, _ = self.encode_and_sample_post(x1)
        f2_mean, _, _, z2_mean, _, _ = self.encode_and_sample_post(x2)

        f_expand1 = f1_mean.expand(-1, self.frames, self.f_dim)
        f_expand2 = f2_mean.expand(-1, self.frames, self.f_dim)

        z1f2 = torch.cat((z1_mean, f_expand2), dim=2)
        z2f1 = torch.cat((z2_mean, f_expand1), dim=2)

        recon_x12 = self.decoder(z1f2)
        recon_x21 = self.decoder(z2f1)

        return recon_x12, recon_x21

    def encode_conv(self, x):
        conv_x = self.encoder_frame(x)  # conv_x: (batch_size, 15, 128)

        # Subtract conv_x first frame from all others conv_x frames
        conv_x_static = conv_x[:, 0][:, None, :]
        conv_x_dynamic = conv_x - conv_x[:, 0][:, None, :]

        return conv_x_static, conv_x_dynamic


class classifier_Sprite_all(nn.Module):
    def __init__(self, opt):
        super(classifier_Sprite_all, self).__init__()
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        from dbse_utils import encoder
        self.encoder = encoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_skin = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_top = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_pant = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_hair = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_action = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 9))

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.cls_action(lstm_out_f), self.cls_skin(lstm_out_f), self.cls_pant(lstm_out_f), \
               self.cls_top(lstm_out_f), self.cls_hair(lstm_out_f)


class classifier_MUG(nn.Module):
    def __init__(self, opt):
        super(classifier_MUG, self).__init__()
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        from dbse_utils import encoder
        self.encoder = encoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_dyn = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_st = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 52))

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.cls_dyn(lstm_out_f), self.cls_st(lstm_out_f)
