import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from video.utils.video_utils import reorder, KL_divergence, inception_score, entropy_Hy, entropy_Hyx

def check_cls(opt, dbse, classifier, test_loader, run):
    """
    Evaluate the classifier model across different labels such as action, skin, pant, top, and hair.

    :param opt: Namespace containing the model configuration and options.
    :param dbse: Model used for evaluation with forward functions for classification.
    :param classifier: The classification model.
    :param test_loader: DataLoader providing batches of test data.
    :param run: Identifier for the current run (not used in this function).

    :return: A tuple of mean accuracy values for action, skin, pant, top, and hair.
    """
    e_values_action, e_values_skin, e_values_pant, e_values_top, e_values_hair = [], [], [], [], []
    for epoch in range(opt.niter):

        print("Epoch", epoch)
        dbse.eval()
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['video']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            if opt.type_gt == "action":
                recon_x_sample, recon_x = dbse.forward_fixed_action_for_classification(x)
            else:
                recon_x_sample, recon_x = dbse.forward_fixed_content_for_classification(x)

            with torch.no_grad():
                pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(x)
                pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)
                pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = classifier(recon_x)

                pred1 = F.softmax(pred_action1, dim=1)
                pred2 = F.softmax(pred_action2, dim=1)
                pred3 = F.softmax(pred_action3, dim=1)

            label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
            label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)
            label2_all.append(label2)

            pred1_all.append(pred1.detach().cpu().numpy())
            pred2_all.append(pred2.detach().cpu().numpy())
            label_gt.append(np.argmax(label_D.detach().cpu().numpy(), axis=1))

            def count_D(pred, label, mode=1):
                return (pred // mode) == (label // mode)

            # action
            acc0_sample = (np.argmax(pred_action2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_D.cpu().numpy(), axis=1)).mean()
            # skin
            acc1_sample = (np.argmax(pred_skin2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 0].cpu().numpy(), axis=1)).mean()
            # pant
            acc2_sample = (np.argmax(pred_pant2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 1].cpu().numpy(), axis=1)).mean()
            # top
            acc3_sample = (np.argmax(pred_top2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 2].cpu().numpy(), axis=1)).mean()
            # hair
            acc4_sample = (np.argmax(pred_hair2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 3].cpu().numpy(), axis=1)).mean()
            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample
            mean_acc2_sample += acc2_sample
            mean_acc3_sample += acc3_sample
            mean_acc4_sample += acc4_sample

        print(
            'Test sample: action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
                mean_acc0_sample / len(test_loader) * 100,
                mean_acc1_sample / len(test_loader) * 100, mean_acc2_sample / len(test_loader) * 100,
                mean_acc3_sample / len(test_loader) * 100, mean_acc4_sample / len(test_loader) * 100))

        label2_all = np.hstack(label2_all)
        label_gt = np.hstack(label_gt)
        pred1_all = np.vstack(pred1_all)
        pred2_all = np.vstack(pred2_all)

        acc = (label_gt == label2_all).mean()
        kl = KL_divergence(pred2_all, pred1_all)

        nSample_per_cls = min([(label_gt == i).sum() for i in np.unique(label_gt)])
        index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        pred2_selected = pred2_all[index]

        IS = inception_score(pred2_selected)
        H_yx = entropy_Hyx(pred2_selected)
        H_y = entropy_Hy(pred2_selected)

        print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc * 100, kl, IS, H_yx, H_y))

        e_values_action.append(mean_acc0_sample / len(test_loader) * 100)
        e_values_skin.append(mean_acc1_sample / len(test_loader) * 100)
        e_values_pant.append(mean_acc2_sample / len(test_loader) * 100)
        e_values_top.append(mean_acc3_sample / len(test_loader) * 100)
        e_values_hair.append(mean_acc4_sample / len(test_loader) * 100)


    return np.mean(e_values_action), np.mean(e_values_skin), np.mean(e_values_pant), np.mean(e_values_top), np.mean(e_values_hair)



