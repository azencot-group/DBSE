import numpy as np


class DbseLoss(object):
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
