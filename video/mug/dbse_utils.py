import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- encoder -----------------------
class dcgan_conv(nn.Module):
    """
    Convolutional block for the DCGAN architecture.

    :param nin: Number of input channels.
    :param nout: Number of output channels.
    :return: Forward pass output.
    """
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    """
    Encoder for a generative model using DCGAN architecture.

    :param dim: Dimension of the latent space.
    :param nc: Number of input channels (default is 1).
    :return: Forward pass output and intermediate states.
    """
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


# ---------------- decoder -----------------------
"""
# Using transpose conv as the block to up-sample
"""
class dcgan_upconv(nn.Module):
    """
    Transposed convolutional block for upsampling in DCGAN.

    :param nin: Number of input channels.
    :param nout: Number of output channels.
    :return: Forward pass output.
    """
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class decoder_convT(nn.Module):
    """
    Decoder using transposed convolution for reconstructing the image.

    :param dim: Dimension of the latent space.
    :param nc: Number of output channels (default is 1).
    :return: Forward pass output.
    """
    def __init__(self, dim, nc=1):
        super(decoder_convT, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output


class decoder_convT_static(nn.Module):
    """
    Static decoder using transposed convolution for reconstructing the image.

    :param dim: Dimension of the latent space.
    :param nc: Number of output channels (default is 1).
    :return: Forward pass output.
    """
    def __init__(self, dim, nc=1):
        super(decoder_convT_static, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], output.shape[1], output.shape[2], output.shape[3])

        return output


"""
# Using bilinear upsampling and conv as the block to up-sample
"""

class upconv(nn.Module):
    """
    Upsampling block using bilinear interpolation and convolution.

    :param nc_in: Number of input channels.
    :param nc_out: Number of output channels.
    :return: Forward pass output.
    """
    def __init__(self, nc_in, nc_out):
        super().__init__()
        self.conv = nn.Conv2d(nc_in, nc_out, 3, 1, 1)
        self.norm = nn.BatchNorm2d(nc_out)

    def forward(self, input):
        out = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        return F.relu(self.norm(self.conv(out)))

class decoder_conv(nn.Module):
    """
    Decoder using a combination of transposed convolution and upsampling.

    :param dim: Dimension of the latent space.
    :param nc: Number of output channels (default is 1).
    :return: Forward pass output.
    """
    def __init__(self, dim, nc=1):
        super(decoder_conv, self).__init__()
        self.dim = dim
        nf = 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(),
            # state size. (nf*8) x 4 x 4
            upconv(nf * 8, nf * 4),
            # state size. (nf*4) x 8 x 8
            upconv(nf * 4, nf * 2),
            # state size. (nf*2) x 16 x 16
            upconv(nf * 2, nf * 2),
            # state size. (nf*2) x 32 x 32
            upconv(nf * 2, nf),
            # state size. (nf) x 64 x 64
            nn.Conv2d(nf, nc, 1, 1, 0),
            nn.Sigmoid()
        )


    def forward(self, input):
        output = self.main(input.view(-1, self.dim, 1, 1))
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output

class decoder_conv_static(nn.Module):
    """
    Static decoder using a combination of transposed convolution and upsampling.

    :param dim: Dimension of the latent space.
    :param nc: Number of output channels (default is 1).
    :return: Forward pass output.
    """
    def __init__(self, dim, nc=1):
        super(decoder_conv_static, self).__init__()
        self.dim = dim
        nf = 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(),
            # state size. (nf*8) x 4 x 4
            upconv(nf * 8, nf * 4),
            # state size. (nf*4) x 8 x 8
            upconv(nf * 4, nf * 2),
            # state size. (nf*2) x 16 x 16
            upconv(nf * 2, nf * 2),
            # state size. (nf*2) x 32 x 32
            upconv(nf * 2, nf),
            # state size. (nf) x 64 x 64
            nn.Conv2d(nf, nc, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.dim, 1, 1))
        output = output.view(input.shape[0], output.shape[1], output.shape[2], output.shape[3])

        return output


class DbseLoss(object):
    """
    Class for tracking and averaging loss values.

    :return: None
    """
    def __init__(self):
        self.reset()

    def update(self, recon_seq, recon_frame, kld_f, kld_z):
        """Update loss values with new reconstructions and KLDs.

        :param recon_seq: Reconstruction loss for sequences.
        :param recon_frame: Reconstruction loss for frames.
        :param kld_f: Kullback-Leibler divergence for frames.
        :param kld_z: Kullback-Leibler divergence for latent space.
        :return: None
        """
        self.recon_seq.append(recon_seq)
        self.recon_frame.append(recon_frame)
        self.kld_f.append(kld_f)
        self.kld_z.append(kld_z)

    def reset(self):
        """Reset the loss values.

        :return: None
        """
        self.recon_seq = []
        self.recon_frame = []
        self.kld_f = []
        self.kld_z = []

