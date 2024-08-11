import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearUnit(nn.Module):
  def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
    super(LinearUnit, self).__init__()
    if batchnorm is True:
      self.model = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features), nonlinearity)
    else:
      self.model = nn.Sequential(
        nn.Linear(in_features, out_features), nonlinearity)

  def forward(self, x):
    return self.model(x)


class decoder(nn.Module):
  def __init__(self, opt):
    super(decoder, self).__init__()

    self.args = opt
    self.lstm = nn.LSTM(opt.f_dim + opt.z_dim, opt.fft_size // 2 + 1, batch_first=True, bias=True,
                        bidirectional=False)
    self.lstm_out_dim = (opt.fft_size // 2 + 1)
    self.dec_net = nn.Sequential(nn.Linear(self.lstm_out_dim, (opt.fft_size // 2 + 1)))

  def forward(self, x):
    output = self.dec_net(self.lstm(x)[0])
    # output = self.dec_net2(x)
    return output


class BatchLinearUnit(nn.Module):
  def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
    super(BatchLinearUnit, self).__init__()
    self.lin = nn.Linear(in_features, out_features)
    self.nrm = nn.BatchNorm1d(out_features)
    self.non_lin = nonlinearity

  def forward(self, x):
    x_lin = self.lin(x)
    x_nrm = self.nrm(x_lin.permute(0, 2, 1))
    return self.non_lin(x_nrm.permute(0, 2, 1))


def reparameterize(mean, logvar, random_sampling=True):
  # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
  if random_sampling is True:
    eps = torch.randn_like(logvar)
    std = torch.exp(0.5 * logvar)
    z = mean + eps * std
    return z
  else:
    return mean


class MLP(nn.Module):
  def __init__(self, input_size, hidden_din):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, input_size // 2)
    self.fc2 = nn.Linear(input_size // 2, input_size // 4)
    self.fc3 = nn.Linear(input_size // 4, hidden_din)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x


class DBSE(nn.Module):
  def __init__(self, opt):
    super(DBSE, self).__init__()
    self.f_dim = opt.f_dim  # content
    self.z_dim = opt.z_dim  # motion
    self.g_dim = opt.fft_size // 2 + 1
    # self.channels = opt.channels  # frame feature
    self.hidden_dim = opt.rnn_size
    self.f_rnn_layers = opt.f_rnn_layers
    self.frames = 20

    # Frame encoder and decoder
    # self.encoder = encoder(self.g_dim, self.channels)
    self.decoder = decoder(opt)

    # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
    self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
    self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

    self.static_linear_proj = nn.Linear(self.g_dim, self.hidden_dim)
    self.tanh = nn.Tanh()

    self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
    self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
    self.z_lstm = nn.LSTM(self.g_dim, self.z_dim, 2, bidirectional=True, batch_first=True)    # self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
    self.f_mean = LinearUnit(self.hidden_dim, self.f_dim, False)
    self.f_logvar = LinearUnit(self.hidden_dim, self.f_dim, False)

    self.z_rnn = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
    # Each timestep is for each z so no reshaping and feature mixing
    self.z_mean = nn.Linear(self.z_dim * 2, self.z_dim)
    self.z_logvar = nn.Linear(self.z_dim * 2, self.z_dim)

    self.mlp_encoder = MLP(4020, self.g_dim)
    self.z_lstm_encoder = torch.nn.LSTM(201, self.g_dim, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(2 * self.g_dim, self.g_dim)

  def encode_and_sample_post(self, x):

    bsz, seq_len, h_dim = x.shape
    enc_x = self.z_lstm_encoder(x)[0]
    enc_x = self.fc(enc_x)

    # get f (anchor is 0, can be changed to random choice):
    conv_x_dynamic = enc_x - enc_x[:, 0][:, None, :]
    # ---- compute static features ----
    kk = 0  # kk = torch.randint(seq_len, (1,)).item()
    f = self.tanh(self.static_linear_proj(enc_x[:, kk]))

    f_mean = self.f_mean(f)
    f_logvar = self.f_logvar(f)
    f_post = reparameterize(f_mean, f_logvar, random_sampling=True)

    # ---- compute dynamics ----
    first_frame_dynamic = reparameterize(torch.zeros((conv_x_dynamic.shape[0], conv_x_dynamic.shape[-1])).cuda(),
                                         torch.zeros((conv_x_dynamic.shape[0], conv_x_dynamic.shape[-1])).cuda(),
                                         random_sampling=True)
    z = torch.cat((first_frame_dynamic[:, None], conv_x_dynamic[:, 1:]), dim=1)
    z = self.z_lstm(z)[0]
    # features, _ = self.z_rnn(output)
    z_mean = self.z_mean(z)
    z_logvar = self.z_logvar(z)
    z_post = reparameterize(z_mean, z_logvar, random_sampling=True)

    # f_mean is list if triple else not
    return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

  def forward(self, x):
    b, t, h = x.shape
    f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
    z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)

    f_expand = f_post.unsqueeze(1).expand(-1, t, self.f_dim)
    z_zeros = torch.zeros_like(z_post)
    zf = torch.cat((z_post, f_expand), dim=2)
    f_stat = torch.cat((z_zeros, f_expand), dim=2)
    recon_seq_x = self.decoder(zf)
    recon_frame_x = self.decoder(f_stat).squeeze()
    return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
           recon_frame_x, recon_seq_x

  def forward_fixed_motion(self, x):
    z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
    f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

    z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
    f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
    zf = torch.cat((z_repeat, f_expand), dim=2)
    recon_x = self.decoder(zf)
    return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

  def forward_fixed_content(self, x):
    b, s, h = x.shape
    z_mean_prior, z_logvar_prior, _ = self.sample_z(b, s, random_sampling=self.training)
    f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

    f_repeat = f_post[0].repeat(f_post.shape[0], 1)
    f_expand = f_repeat.unsqueeze(1).expand(-1, s, self.f_dim)

    zf = torch.cat((z_post, f_expand), dim=2)
    recon_x = self.decoder(zf)
    return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

  def forward_fixed_content_for_classification(self, x):
    b, s, h = x.shape
    z_mean_prior, z_logvar_prior, _ = self.sample_z(b, s, random_sampling=True)
    f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

    f_expand = f_mean.unsqueeze(1).expand(-1, s, self.f_dim)

    zf = torch.cat((z_mean_prior, f_expand), dim=2)
    recon_x_sample = self.decoder(zf)

    zf = torch.cat((z_mean_post, f_expand), dim=2)
    recon_x = self.decoder(zf)

    return recon_x_sample, recon_x

  def forward_fixed_motion_for_multi_pos_v2(self, x):
    b, s, h = x.shape
    z_mean_prior, z_logvar_prior, _ = self.sample_z(b, s, random_sampling=True)
    f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

    f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                  random_sampling=True)
    f_expand = f_prior.unsqueeze(1).expand(-1, s, self.f_dim)
    zf = torch.cat((z_mean_post, f_expand), dim=2)
    recon_x_sample = self.decoder(zf)

    f_expand = f_mean.unsqueeze(1).expand(-1, s, self.f_dim)
    zf = torch.cat((z_mean_post, f_expand), dim=2)
    recon_x = self.decoder(zf)

    return recon_x_sample, recon_x

  def forward_fixed_motion_for_classification(self, x):
    b, s, h = x.shape
    z_mean_prior, z_logvar_prior, _ = self.sample_z(b, s, random_sampling=True)
    f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

    f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                  random_sampling=True)
    f_expand = f_prior.unsqueeze(1).expand(-1, s, self.f_dim)
    zf = torch.cat((z_mean_post, f_expand), dim=2)
    recon_x_sample = self.decoder(zf)

    f_expand = f_mean.unsqueeze(1).expand(-1, s, self.f_dim)
    zf = torch.cat((z_mean_post, f_expand), dim=2)
    recon_x = self.decoder(zf)

    return recon_x_sample, recon_x

  def encoder_frame(self, x):
    # input x is list of length Frames [batchsize, channels, size, size]
    x_shape = x.shape
    x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
    x_embed = self.encoder(x)[0]
    # to [batch_size , frames, embed_dim]
    return x_embed.view(x_shape[0], x_shape[1], -1)

  def reparameterize(self, mean, logvar, random_sampling=True):
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
      eps = torch.randn_like(logvar)
      std = torch.exp(0.5 * logvar)
      z = mean + eps * std
      return z
    else:
      return mean

  def sample_z_prior_test(self, n_sample, n_frame, random_sampling=True):
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
      z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
      if z_out is None:
        # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
        z_out = z_prior.unsqueeze(1)
        z_means = z_mean_t.unsqueeze(1)
        z_logvars = z_logvar_t.unsqueeze(1)
      else:
        # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
        z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
        z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
        z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        # z_t = z_post[:,i,:]
      z_t = z_prior
    return z_means, z_logvars, z_out

  def sample_z_prior_train(self, z_post, random_sampling=True):
    z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
    z_means = None
    z_logvars = None
    batch_size, seq_len, h = z_post.shape

    z_t = torch.zeros(batch_size, self.z_dim).cuda()
    h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
    c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
    h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
    c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

    for i in range(seq_len):
      # two layer LSTM and two one-layer FC
      h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
      h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

      z_mean_t = self.z_prior_mean(h_t_ly2)
      z_logvar_t = self.z_prior_logvar(h_t_ly2)
      z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
      if z_out is None:
        # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
        z_out = z_prior.unsqueeze(1)
        z_means = z_mean_t.unsqueeze(1)
        z_logvars = z_logvar_t.unsqueeze(1)
      else:
        # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
        z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
        z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
        z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
      z_t = z_post[:, i, :]
    return z_means, z_logvars, z_out

  # If random sampling is true, reparametrization occurs else z_t is just set to the mean
  def sample_z(self, batch_size, seq_len, random_sampling=True):
    z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
    z_means = None
    z_logvars = None

    # All states are initially set to 0, especially z_0 = 0
    z_t = torch.zeros(batch_size, self.z_dim).cuda()
    # z_mean_t = torch.zeros(batch_size, self.z_dim)
    # z_logvar_t = torch.zeros(batch_size, self.z_dim)
    h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
    c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
    h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
    c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
    for _ in range(seq_len):
      # h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
      # two layer LSTM and two one-layer FC
      h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
      h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

      z_mean_t = self.z_prior_mean(h_t_ly2)
      z_logvar_t = self.z_prior_logvar(h_t_ly2)
      z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
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
      # two layer LSTM and two one-layer FC
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
