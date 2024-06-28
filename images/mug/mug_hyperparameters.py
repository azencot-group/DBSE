import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--nEpoch', default=2000, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval', default=50, type=int, help='evaluate every n epoch')
parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')
parser.add_argument('--neptune', default=True, type=bool, help='activate neptune tracking')

parser.add_argument('--dataset', default='MUG', type=str, help='dataset to train')
parser.add_argument('--frames', default=15, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--decoder', default='ConvT', type=str, help='Upsampling+Conv or Transpose Conv: Conv or ConvT')

parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
parser.add_argument('--f_dim', default=64, type=int, help='dim of f')
parser.add_argument('--z_dim', default=64, type=int, help='dimensionality of z_t')
parser.add_argument('--g_dim', default=128, type=int,
                    help='dimensionality of encoder output vector and decoder input vector')

parser.add_argument('--loss_recon', default='L2', type=str, help='reconstruction loss: L1, L2')
parser.add_argument('--note', default='LogNCELoss', type=str, help='appx note')
parser.add_argument('--weight_f', default=1, type=float, help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', default=1, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_rec_seq', default=90, type=float, help='weighting on reconstruction')
parser.add_argument('--weight_rec_frame', default=30, type=float, help='weighting on reconstruction')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--sche', default='const', type=str, help='scheduler')

parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
parser.add_argument('--niter', type=int, default=5, help='number of runs for testing')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
