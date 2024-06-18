import torch
import torch.nn as nn
import torch.distributions as dist

def reparameterize(mean, logvar, random_sampling=True):
    # Re-parametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar) # eps ~ N(0, 1) in size of logvar tensor
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        """Initializes the instance"""
        super(Encoder, self).__init__()
        self.hidden_sizes = hidden_sizes # [32, 64, 32]

        # Construct the layers list starting from the input size
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x is in batch x time x feature
        batch_size, time_steps, features = x.size()

        # Reshape the tensor to [-1, features] for applying the Linear layers
        x = x.view(-1, features)
        x = self.fc(x)

        # Reshape back to [batch, time, new_feature_size]
        x = x.view(batch_size, time_steps, -1)
        return x


class Decoder(nn.Module):
    def __init__(self, args, output_size, output_length, hidden_sizes):
        """Initializes the instance"""
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.output_length = output_length
        self.hidden_sizes = hidden_sizes

        self.f_dim = args.f_dim                                 # dim of latent static
        self.z_dim = args.z_dim

        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(self.f_dim + self.z_dim, hidden_sizes[0]),
            nn.Tanh()
        )

        # Fully Connected layers
        fc_layers = []
        for i in range(1, len(hidden_sizes)):
            fc_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            fc_layers.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_layers[:-1])  # Remove the last ReLU

        # Recurrent Layer
        self.rnn = nn.LSTM(hidden_sizes[0], hidden_sizes[0], batch_first=True)

        # Mean Generator
        self.mean_gen = nn.Linear(hidden_sizes[-1], self.output_size)

        # Covariance Generator
        self.cov_gen = nn.Sequential(
            nn.Linear(hidden_sizes[-1], self.output_size),
            nn.Sigmoid()
        )

    def forward(self, z_t, static=False):
        """Reconstruct a time series from the representations of all windows over time

        Args:
            z_t: Representation of signal windows with shape [batch_size, n_windows, representation_size] = [300, 20, 16]
        """
        n_batch, prior_len, feature_dim = z_t.shape # n_batch = 300, prior_len = 20, feature_dim = 16
        z_reshaped = z_t.view(-1, feature_dim)
        emb = self.embedding(z_reshaped) # emb = [300*20 = 6000, 16]
        emb = emb.view(n_batch, prior_len, -1) # emb = [300, 20, 16]
        recon_seq = []

        for t in range(prior_len):
            h_0 = emb[:, t, :].unsqueeze(0).contiguous()  # Hidden state h_0.shape = [1, 300, 32]
            c_0 = emb[:, t, :].unsqueeze(0).contiguous()  # Cell state c_0.shape = [1, 300, 32]

            if static:
                rnn_out, _ = self.rnn(
                    torch.randn(n_batch, 1, self.hidden_sizes[0], device=z_t.device),
                    (h_0, c_0)
                )  # rnn_out.shape = [300, 4, 32]
            else:
                rnn_out, _ = self.rnn(
                    torch.randn(n_batch, self.output_length, self.hidden_sizes[0], device=z_t.device),
                    (h_0, c_0)
                ) # rnn_out.shape = [300, 4, 32]
            recon_seq.append(rnn_out)

        recon_seq = torch.cat(recon_seq, 1)
        recon_seq = self.fc(recon_seq)
        x_mean = self.mean_gen(recon_seq)
        x_cov = self.cov_gen(recon_seq)
        return dist.Normal(loc=x_mean, scale=x_cov*0.5)

class DBSE(nn.Module):
    def __init__(self, encoder, decoder, configs, args):
        """Initializes the instance"""
        super(DBSE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.feature_dim = configs["feature_size"]              # dim of features (10 physionet, 10 air_quailty)
        self.sample_len = configs["t_len"]                      # dim of time (60 physionet, 672 air_quality)
        self.window_size = configs["window_size"]               # split time series to windows of this size
        self.dataset_size = 1

        self.fc_dim = encoder.hidden_sizes[-1]                  # dim of FC encoder/decoder output/input
        self.hidden_dim = 32                                    # dim of latent information
        self.f_dim = args.f_dim                                 # dim of latent static
        self.z_dim = args.z_dim                                 # dim of latent dynamic

        self.M = configs["mc_samples"]                          # used for tiling information ??
        self.weight_rec = args.weight_rec
        self.weight_rec_stat = args.weight_rec_stat
        self.weight_f = args.weight_f
        self.weight_z = args.weight_z
        self.weight_mi = args.weight_mi

        self.g_dim = args.g_dim
        self.hidden_dim = args.rnn_size

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(input_size=self.z_dim, hidden_size=self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # ----- Posterior of content and motion
        # content and motion features share one bi-lstm
        self.z_lstm = torch.nn.LSTMCell(self.g_dim, self.hidden_dim)

        # Posterior of static and dynamic
        # static and dynamic features share one lstm
        self.fz_lstm = nn.LSTM(input_size=self.fc_dim, hidden_size=self.fc_dim, batch_first=True)
        self.f_mean = nn.Linear(self.fc_dim, self.f_dim)
        self.f_logvar = nn.Linear(self.fc_dim, self.f_dim)

        # ----- Static features
        self.static_feature_extractor = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim),  # First linear layer
            nn.Tanh()
        )


        # for f static
        self.gru = nn.GRU(input_size=self.fc_dim, hidden_size=self.fc_dim, batch_first=True)

        # dynamic features from the next lstm
        self.z_lstm = nn.LSTM(input_size=self.fc_dim, hidden_size=self.fc_dim, batch_first=True)
        self.z_mean = nn.Linear(self.fc_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.fc_dim, self.z_dim)

        # dynamic features from the next lstm
        self.z_lstm = nn.LSTMCell(self.fc_dim, self.fc_dim)
        self.z_mean = nn.Linear(self.fc_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.fc_dim, self.z_dim)



    def forward(self, x_seq, x_lens, mask_seq=None):
        # encode and sample post
        f_mean, f_logvar, f_post, z_mean, z_logvar, z_post = self.encode_and_sample_post(x_seq, mask_seq, x_lens, self.window_size, random_sampling=True)

        # sample prior
        z_prior_mean, z_prior_logvar, z_prior = self.sample_motion_prior_train(z_post, random_sampling=True) # z_prior.shape = (400, 20, 4)

        f_expand = f_post.repeat(1, z_post.shape[1], 1)
        zf = torch.cat([z_post, f_expand], dim=2)
        zf_stat = torch.cat((z_post[:, 0][:, None], f_post), dim=2)


        px_hat_stat = self.decoder(zf_stat, static=True)
        recon_x_frame = px_hat_stat.rsample()

        px_hat = self.decoder(zf)
        recon_x = px_hat.rsample()

        f_post = f_post.squeeze(1)
        f_logvar = f_logvar.squeeze(1)
        f_mean = f_mean.squeeze(1)

        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x, px_hat, recon_x_frame, px_hat_stat

    def encode_and_sample_post(self, x, mask, x_lens, window_size, random_sampling):
        # pipeline: x (fc_enc)-> fc_x (z_lstm, z_rnn)-> (f_post, z_post)

        bsz, seq_len, h_dim = x.shape

        # handle mask
        if not mask is None:
            mask = (torch.sum(mask, dim=-1) < int(0.7 * x.shape[-1])) # mask.shape = [150, 672]

        # FC encoder
        fc_x = self.encoder(x) # fc_x.shape = [150, 672, 32]
        x_encoded_static = fc_x[:, 0: window_size] # x_encoded_static.shape = [150, 24, 32]
        x_encoded_static_expand = x_encoded_static.repeat(1, fc_x.shape[1] // window_size, 1)
        x_encoded_dynamic = fc_x - x_encoded_static_expand

        # Static feature extractor
        _, h = self.gru(x_encoded_static)
        h = h.view(bsz, 1, -1)
        f = self.static_feature_extractor(h)

        # Re parameterize the static features
        f_mean = self.f_mean(f)
        f_logvar = self.f_logvar(f)
        f_post = reparameterize(f_mean, f_logvar, random_sampling=True)

        first_frame_dynamic = reparameterize(torch.zeros((x_encoded_dynamic.shape[0], x_encoded_dynamic.shape[-1])).cuda(),
                                             torch.zeros((x_encoded_dynamic.shape[0], x_encoded_dynamic.shape[-1])).cuda(),
                                             random_sampling=True).unsqueeze(1).repeat(1, window_size, 1)

        z = torch.cat((first_frame_dynamic, fc_x[:, window_size:]), dim=1)

        # LSTM encoder (static)
        dynamic_output = []
        for t in range(0, x.shape[1] - window_size + 1, window_size): # x.shape[1] = 80, window_size = 4
            if mask is not None:
                lstm_outputs, _ = self.fz_lstm(z[:, t:t + window_size, :] * mask[:, t:t + window_size].unsqueeze(-1))
                x_mapped = lstm_outputs[:, -1, :]
            else:
                lstm_outputs, _ = self.fz_lstm(z[:, t:t + window_size, :])
                x_mapped = lstm_outputs[:, -1, :]
            dynamic_output.append(x_mapped)

        # Stack along the new axis
        dynamic_output = torch.stack(dynamic_output, dim=1)

        z_mean = self.z_mean(dynamic_output)
        z_logvar = self.z_logvar(dynamic_output)
        z_post = reparameterize(z_mean, z_logvar, random_sampling=True)


        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

    # ------ sample z from learned LSTM prior base on previous postior, teacher forcing for training  ------
    def sample_motion_prior_train(self, z_post, random_sampling=True):
        # z_post is in batch x (new) time x features
        z_out = None
        z_means = None
        z_logvars = None
        batch_size, frames = z_post.shape[0], z_post.shape[1] # batch_size = 300, frames = 20

        z_t = torch.zeros(batch_size, self.z_dim).cuda() # z_t.shape = [300, 4]
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda() # h_t_ly1.shape = [300, 32]
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda() # c_t_ly1.shape = [300, 32]
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda() # h_t_ly2.shape = [300, 32]
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda() # c_t_ly2.shape = [300, 32]

        for i in range(frames):
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


    def get_trainable_vars(self):
        self.compute_loss(x=tf.random.normal(shape=(1, self.sample_len, self.feature_dim), dtype=tf.float32),
                          m_mask=tf.zeros(shape=(1, self.sample_len, self.feature_dim), dtype=tf.float32))
        return self.trainable_variables