import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import shutil
from torch.utils.data import DataLoader
from mug_data_class import MugDataset, image_transforms
from dbse_model import classifier_MUG

### ------------------------------------ Consts ----------------------------------------------- ###
MUG_TEST_DATASET_PATH = '/cs/cs_groups/azencot_group/datasets/mug_pre2_test'
MUG_TRAIN_DATASET_PATH = '/cs/cs_groups/azencot_group/datasets/mug_pre2_train'
CLS_PATH = '/cs/cs_groups/azencot_group/mutual_information_disentanglement/classifiers/mug_cls_new_contrastive.tar'


def load_dataset(opt, mode='train'):
    dataset_tr = MugDataset(MUG_TRAIN_DATASET_PATH, transform=image_transforms)
    dataset_te = MugDataset(MUG_TEST_DATASET_PATH, transform=image_transforms)
    train_loader = DataLoader(dataset_tr, batch_size=opt.batch_size, drop_last=True, num_workers=4, shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(dataset_te, batch_size=opt.batch_size, drop_last=True, num_workers=4, shuffle=True,
                             pin_memory=True)

    bad_samples = [514, 517, 518, 519, 520, 542, 543, 32, 36, 550, 551, 557, 562, 52, 571, 573, 67,
                   76, 628, 630, 135, 143, 659, 660, 665, 685, 177, 689, 691, 194,
                   706, 708, 720, 721, 213, 729, 745, 236, 338, 371, 404, 429,
                   434, 436, 437, 439, 440, 441, 447, 449, 461, 462, 464, 466, 483, 484, 489, 491, 494, 501, 510]

    print(len(test_loader.dataset.videos))

    test_loader.dataset.videos = [i for j, i in enumerate(test_loader.dataset.videos) if j not in bad_samples]

    print(len(test_loader.dataset.videos))

    opt.dataset_size = len(dataset_tr)
    if mode == 'eval':
        return test_loader
    return test_loader, train_loader


def define_classifier(opt):
    old_rnn_size = opt.rnn_size
    old_g_dim = opt.g_dim
    opt.rnn_size = 256
    opt.g_dim = 128
    classifier = classifier_MUG(opt)
    opt.cls_path = CLS_PATH
    loaded_dict = torch.load(opt.cls_path)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()
    opt.rnn_size = old_rnn_size
    opt.g_dim = old_g_dim
    return classifier



