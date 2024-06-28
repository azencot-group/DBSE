import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch.optim as optim
import torch.utils.data
from dbse_model import DBSE
from utils import define_seed
from mug_utils_train import *
from mug_hyperparameters import *
from mug_utils import *
from neptune_utils.neptune_utils import *

### ------------------------------------ Consts ----------------------------------------------- ###
SAVER_MODEL_PATH = '/home/arbivid/no_mi_mug_5/mug/no_mi/sbatch/logs/model2000a_action=87.01704545454547a_subj=3.778409090909091s_action=21.392045454545457s_subj=99.51704545454545.pth'
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjODljNGI3NS0yOWYyLTRhN2QtOWVmYS1iMTE4ODAxYWM5NmQifQ=="
NEPTUNE_PROJECT_NAME = "azencot-group/No-Mi-Experiments"
NEPTUNE_TAGS = ["MUG", "Evaluation"]

def pre_eval(opt):
    define_seed(opt)

    # Optimizers and model define
    opt.optimizer = optim.Adam
    model = DBSE(opt)

    # Load the saved model parameters
    checkpoint = torch.load(SAVER_MODEL_PATH)
    model.load_state_dict(checkpoint['model'])

    # model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    model = model.cuda()

    # Load classifier for testing during train
    classifier = define_classifier(opt)

    # Load a dataset
    test_loader = load_dataset(opt, 'eval')

    run = None
    if opt.neptune:
        run = init_neptune(opt, NEPTUNE_PROJECT_NAME, NEPTUNE_API_TOKEN, NEPTUNE_TAGS)

    return classifier, model, optimizer, test_loader, run


def eval_model(opt):
    classifier, model, optimizer, test_loader, run = pre_eval(opt)
    # Set model to eval mode
    model.eval()
    opt.type_gt = 'action'
    a_action, a_subj, matrix = check_cls_mug(opt, model, classifier, test_loader, run)
    opt.type_gt = 'aaction'
    s_action, s_subj = check_cls_mug(opt, model, classifier, test_loader, run)
    print(f"a_action: {a_action}, a_subj: {a_subj}")
    print(f"s_action: {s_action}, s_subj: {s_subj}")

if __name__ == '__main__':

    eval_model(opt)
