import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--nEpoch', default=500, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')
parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')

parser.add_argument('--dataset', default='TIMIT', type=str, help='dataset to train')
parser.add_argument("--dataset_path", default=None)
parser.add_argument("--train_annotation", default=None)
parser.add_argument("--valid_annotation", default=None)
parser.add_argument("--test_annotation", default=None)
parser.add_argument('--decoder', default='ConvT', type=str, help='Upsampling+Conv or Transpose Conv: Conv or ConvT')

parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
parser.add_argument('--f_dim', default=128, type=int, help='dim of f')
parser.add_argument('--z_dim', default=128, type=int, help='dimensionality of z_t')
parser.add_argument('--g_dim', default=128, type=int, help='dimensionality of encoder output vector and decoder input vector')

# audio args
parser.add_argument('--w_len', type=int, default=320, help='window length')
parser.add_argument('--h_len', type=int, default=165, help='hop length')
parser.add_argument('--power', type=int, default=1, help='power of the spectrogram')
parser.add_argument('--fft_size', type=int, default=400, help='power of the spectrogram')

parser.add_argument('--loss_recon', default='L2', type=str, help='reconstruction loss: L1, L2')
parser.add_argument('--note', default='mug', type=str, help='appx note')
parser.add_argument('--weight_f', default=1, type=float, help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', default=1, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_rec', default=20, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_rec_frame', default=100, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--sche', default='const', type=str, help='scheduler')
parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
parser.add_argument('--tag', type=str, default='koopman')
parser.add_argument('--niter', type=int, default=5, help='number of runs for testing')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
