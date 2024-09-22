import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch.optim as optim
import torch.utils.data
from dbse_model import DBSE
from utils import define_seed
from mug_utils_train import *
from mug_hyperparameters import *
from mug_utils import *

# Constants to be defined by the user
SAVER_MODEL_PATH = None
RUN = None

def pre_eval(opt):
    """
    Prepares the model and classifier for evaluation.

    :param opt: Options or configuration settings for the model.
    :return: Tuple containing the classifier, model, optimizer, test data loader, and run configuration.
    """
    define_seed(opt)
    run = RUN
    opt.optimizer = optim.Adam
    model = DBSE(opt)
    checkpoint = torch.load(SAVER_MODEL_PATH)
    model.load_state_dict(checkpoint['model'])
    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    model = model.cuda()
    classifier = define_classifier(opt)
    test_loader = load_dataset(opt, 'eval')
    return classifier, model, optimizer, test_loader, run


def eval_model(opt):
    """
    Evaluates the model's performance on the test dataset.

    :param opt: Options or configuration settings for the model.
    :return: None
    """
    classifier, model, optimizer, test_loader, run = pre_eval(opt)
    model.eval()
    opt.type_gt = 'action'
    a_action, a_subj, matrix = check_cls_mug(opt, model, classifier, test_loader, run)
    opt.type_gt = 'aaction'
    s_action, s_subj = check_cls_mug(opt, model, classifier, test_loader, run)
    print(f"a_action: {a_action}, a_subj: {a_subj}")
    print(f"s_action: {s_action}, s_subj: {s_subj}")


if __name__ == '__main__':
    eval_model(opt)
