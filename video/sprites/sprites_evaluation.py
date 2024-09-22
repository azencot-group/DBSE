import random
import utils

import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from sprites_utils_train import *
from dbse_model import DBSE, classifier_Sprite_all
from sprites_hyperparameters import *

mse_loss = nn.MSELoss().cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

# Constants to be defined by the user
SAVER_MODEL_PATH = None

def eval_model(opt):
    """
    Prepares the model and classifier for evaluation and runs classification checks for both
    'action' and 'aaction' ground truth types.

    :param opt: Namespace containing model configuration and options.
    """
    cdsvae, classifier, test_loader = pre_eval(opt)
    opt.type_gt = 'action'
    check_cls(opt, cdsvae, classifier, test_loader, None)
    opt.type_gt = 'aaction'
    check_cls(opt, cdsvae, classifier, test_loader, None)


def pre_eval(opt):
    """
    Prepares the environment by setting random seeds, loading the model and classifier, and
    initializing data loaders for the dataset.

    :param opt: Namespace containing model configuration and options.
    :return: Tuple of (cdsvae model, classifier, test_loader).
    """
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    # load model
    cdsvae = DBSE(opt).cuda()
    cdsvae.eval()
    classifier = classifier_Sprite_all(opt)
    loaded_dict = torch.load(SAVER_MODEL_PATH)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()
    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)
    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=opt.batch_size,  # 128
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=opt.batch_size,  # 128
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)
    opt.dataset_size = len(train_data)
    return cdsvae, classifier, test_loader


if __name__ == '__main__':
    eval_model(opt)
