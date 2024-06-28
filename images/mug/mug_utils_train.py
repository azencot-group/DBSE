import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from utils import reorder, KL_divergence, inception_score, entropy_Hy, entropy_Hyx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from neptune_utils.neptune_utils import log_to_neptune, upload_image_to_neptune
from PIL import Image
import itertools

action_labels_dic = {
    0: 'anger',
    1: 'disgust',
    2: 'happiness',
    3: 'fear',
    4: 'sadness',
    5: 'surprise',
}
static_labels_dic = {i: str(i) for i in range(51)}


def count_D(pred, label, mode=1):
    return (pred // mode) == (label // mode)


def check_cls_sprites(opt, dbse, classifier, test_loader, run):
    e_values_action, e_values_skin, e_values_pant, e_values_top, e_values_hair = [], [], [], [], []
    for epoch in range(opt.niter):

        print("Epoch", epoch)
        dbse.eval()
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            if opt.type_gt == "action":
                recon_x_sample, recon_x = dbse.forward_fixed_action_for_classification(x)
            else:
                recon_x_sample, recon_x = dbse.forward_fixed_content_for_classification(x)

            with torch.no_grad():
                pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(x)
                pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)

                pred1 = F.softmax(pred_action1, dim=1)
                pred2 = F.softmax(pred_action2, dim=1)

            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
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

    if run and opt.type_gt == "action":
        # action acc should be high, other low
        run['action/a_action'].log(np.mean(e_values_action))
        run['action/a_skin'].log(np.mean(e_values_skin))
        run['action/a_pant'].log(np.mean(e_values_pant))
        run['action/a_top'].log(np.mean(e_values_top))
        run['action/a_hair'].log(np.mean(e_values_hair))

    elif run:
        # action acc should be low, other high
        run['static/s_action'].log(np.mean(e_values_action))
        run['static/s_skin'].log(np.mean(e_values_skin))
        run['static/s_pant'].log(np.mean(e_values_pant))
        run['static/s_top'].log(np.mean(e_values_top))
        run['static/s_hair'].log(np.mean(e_values_hair))

    return np.mean(e_values_action), np.mean(e_values_skin), np.mean(e_values_pant), np.mean(e_values_top), np.mean(
        e_values_hair)


def check_cls_mug(opt, model, classifier, test_loader, run):
    e_values_action, e_values_subj = [], []
    matrix = np.zeros((6, 6))
    for epoch in range(4):

        print("Epoch", epoch)
        model.eval()
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = data[0].cuda(), data[1].cuda(), data[
                2].cuda()  # x: (batch_size, 15, 3, 64, 64) label_A: (batch_size), label_D: (batch_size)

            if opt.type_gt == "action":
                recon_x_sample, recon_x = model.forward_fixed_action_for_classification(
                    x)  # recon_x_sample: (batch_size, 15, 3, 64, 64), recon_x: (batch_size, 15, 3, 64, 64)
            else:
                recon_x_sample, recon_x = model.forward_fixed_content_for_classification(x)

            with torch.no_grad():
                pred_action1, pred_sub1 = classifier(x)  # pred_action1: (batch_size, 6), pred_sub1: (batch_size, 52)
                pred_action2, pred_sub2 = classifier(
                    recon_x_sample)  # pred_action2: (batch_size, 6), pred_sub2: (batch_size, 52)

                pred1 = F.softmax(pred_action1, dim=1)  # pred1: (batch_size, 6)
                pred2 = F.softmax(pred_action2, dim=1)  # pred2: (batch_size, 6)

            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)  # label2: (batch_size)
            label2_all.append(label2)

            pred1_all.append(pred1.detach().cpu().numpy())
            pred2_all.append(pred2.detach().cpu().numpy())
            label_gt.append(label_D.detach().cpu().numpy())

            acc0_sample = count_D(np.argmax(pred_action2.detach().cpu().numpy(), axis=1), label_A.cpu().numpy()).mean()
            for i, j in zip(label_A.cpu().numpy(), np.argmax(pred_action2.detach().cpu().numpy(), axis=1)):
                matrix[i, j] += 1
            acc1_sample = (np.argmax(pred_sub2.detach().cpu().numpy(), axis=1) == label_D.cpu().numpy()).mean()

            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample

        print(
            'Test sample: action_Acc: {:.2f}% subj_Acc: {:.2f}%'.format(
                mean_acc0_sample / len(test_loader) * 100,
                mean_acc1_sample / len(test_loader) * 100))

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
        e_values_subj.append(mean_acc1_sample / len(test_loader) * 100)

    if run and opt.type_gt == "action":
        # action acc should be high, other low
        run['action/a_action'].log(np.mean(e_values_action))
        run['action/a_subj'].log(np.mean(e_values_subj))

    elif run:
        # action acc should be low, other high
        run['static/s_action'].log(np.mean(e_values_action))
        run['static/s_subj'].log(np.mean(e_values_subj))

    return np.mean(e_values_action), np.mean(e_values_subj)


def pair_iterable(iterable):
    """Create pairs from iterable."""
    iterable = iter(iterable)
    return zip(iterable, iterable)


def test_swap(opt, model, classifier, test_loader, run, type='test'):
    values_motion_swap, values_static_swap = [], []
    for epoch in range(opt.niter):
        print("Epoch", epoch)
        model.eval()
        mean_acc_motion, mean_acc_static = 0, 0
        total_samples = 0

        for i, (data1, data2) in enumerate(pair_iterable(test_loader)):
            x1, x1_action_labels, x1_static_labels = data1[0].cuda(), data1[1].cuda(), data1[2].cuda()
            x2, x2_action_labels, x2_static_labels = data2[0].cuda(), data2[1].cuda(), data2[2].cuda()
            total_samples += x1.shape[0] + x2.shape[0]

            with torch.no_grad():
                # recon_z1f2 = The motion of x1 with the content of x2. recon_z2f1 = The motion of x2 with the content of x1
                recon_z1f2, recon_z2f1 = model.forward_for_swap_dynamic_and_static(x1, x2)

                # Obtain predictions by classifier
                pred_z1, pred_f2 = classifier(recon_z1f2)
                pred_z2, pred_f1 = classifier(recon_z2f1)

            pred_z1_label = np.argmax(pred_z1.detach().cpu().numpy(), axis=1)
            pred_f2_label = np.argmax(pred_f2.detach().cpu().numpy(), axis=1)
            pred_z2_label = np.argmax(pred_z2.detach().cpu().numpy(), axis=1)
            pred_f1_label = np.argmax(pred_f1.detach().cpu().numpy(), axis=1)

            mean_acc_motion += np.sum(pred_z1_label == x1_action_labels.cpu().numpy()) + np.sum(
                pred_z2_label == x2_action_labels.cpu().numpy())
            mean_acc_static += np.sum(pred_f1_label == x1_static_labels.cpu().numpy()) + np.sum(
                pred_f2_label == x2_static_labels.cpu().numpy())

        mean_acc_motion /= total_samples
        mean_acc_static /= total_samples

        values_motion_swap.append(mean_acc_motion * 100)
        values_static_swap.append(mean_acc_static * 100)

    if run:
        if type == "train":
            log_to_neptune(run, {type + '/swap/acc_motion': np.mean(values_motion_swap),
                                 type + '/swap/acc_static': np.mean(values_static_swap)})
        else:
            log_to_neptune(run, {'swap/acc_motion': np.mean(values_motion_swap),
                                 'swap/acc_static': np.mean(values_static_swap)})


def arg_max(pred):
    return np.argmax(pred.detach().cpu().numpy(), axis=1)


def get_labels(classifier, data):
    with torch.no_grad():
        pred_action, pred_subject = classifier(data)
    return arg_max(pred_action), arg_max(pred_subject)


def create_dict(name, orig_action, pred_action, orig_subject, pred_subject):
    return {'Img': name,
            'A': action_labels_dic[orig_action],
            'Pred A': action_labels_dic[pred_action],
            'S': orig_subject,
            'Pred S': pred_subject,
            'A_correct?': pred_action == orig_action,
            'S_correct?': pred_subject == orig_subject}


def save_image_helper(sequence_of_images, title_dictionary, image_name, directory_path):
    fig = plt.figure(figsize=(15, 1))
    for j, frame in enumerate(sequence_of_images):
        plt.subplot(1, sequence_of_images.shape[0], j + 1)
        numpy_frame = frame.detach().cpu().numpy()
        plt.imshow(numpy_frame.transpose((1, 2, 0)))  # assuming the image data is in CHW format
        plt.axis('off')
    plt.suptitle('; '.join([f'{key}: {value}' for key, value in title_dictionary.items()]))
    fig.savefig(os.path.join(directory_path, f'{image_name}.png'))
    plt.close(fig)


def merge_images(directory, output_path):
    # Get all image files in the directory
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if
                   file.endswith(('.jpg', '.jpeg', '.png'))]

    image_files.sort()

    # Load images
    images = [Image.open(img) for img in image_files]

    # Make sure images got same width
    max_width = max(image.size[0] for image in images)
    images = [image.resize((max_width, int(max_width * image.size[1] / image.size[0])), Image.ANTIALIAS) for image in
              images]

    # Get total height
    total_height = sum(image.size[1] for image in images)

    # Create new image with width and height
    new_img = Image.new('RGB', (max_width, total_height))

    # Set paste positions
    y_offset = 0
    for image in images:
        new_img.paste(image, (0, y_offset))
        y_offset += image.height

    # Save the image
    new_img.save(output_path)

    # Return the absolute path of the saved image
    return os.path.abspath(output_path)


def test_swap_table(opt, model, classifier, test_loader, run, type='test'):
    values_motion_swap, values_static_swap = [], []
    matrix_predicted_labels = np.zeros((6, 6))
    for epoch in range(opt.niter):
        print("Epoch", epoch)
        model.eval()
        mean_acc_motion, mean_acc_static = 0, 0
        total_samples = 0

        for i, (data1, data2) in enumerate(pair_iterable(test_loader)):
            x1, x1_action_labels, x1_static_labels = data1[0].cuda(), data1[1].cuda(), data1[2].cuda()
            x2, x2_action_labels, x2_static_labels = data2[0].cuda(), data2[1].cuda(), data2[2].cuda()
            total_samples += x1.shape[0] + x2.shape[0]

            with torch.no_grad():
                # recon_z1f2 = The motion of x1 with the content of x2. recon_z2f1 = The motion of x2 with the content of x1
                recon_z1f2, recon_z2f1 = model.forward_for_swap_dynamic_and_static(x1, x2)

                # Obtain predictions by classifier
                pred_z1, pred_f2 = classifier(recon_z1f2)
                pred_z2, pred_f1 = classifier(recon_z2f1)

            pred_z1_label = np.argmax(pred_z1.detach().cpu().numpy(), axis=1)
            pred_f2_label = np.argmax(pred_f2.detach().cpu().numpy(), axis=1)
            pred_z2_label = np.argmax(pred_z2.detach().cpu().numpy(), axis=1)
            pred_f1_label = np.argmax(pred_f1.detach().cpu().numpy(), axis=1)

            for true_label, predicted_label in zip(x1_action_labels, pred_z1_label):
                matrix_predicted_labels[predicted_label][true_label] = matrix_predicted_labels[predicted_label][
                                                                           true_label] + 1
            for true_label, predicted_label in zip(x2_action_labels, pred_z2_label):
                matrix_predicted_labels[predicted_label][true_label] = matrix_predicted_labels[predicted_label][
                                                                           true_label] + 1

            mean_acc_motion += np.sum(pred_z1_label == x1_action_labels.cpu().numpy()) + np.sum(
                pred_z2_label == x2_action_labels.cpu().numpy())
            mean_acc_static += np.sum(pred_f1_label == x1_static_labels.cpu().numpy()) + np.sum(
                pred_f2_label == x2_static_labels.cpu().numpy())

        mean_acc_motion /= total_samples
        mean_acc_static /= total_samples

        values_motion_swap.append(mean_acc_motion * 100)
        values_static_swap.append(mean_acc_static * 100)

    distributions = np.zeros((6, 6))
    for column in range(6):
        # Get the sum of the column
        column_sum = np.sum(matrix_predicted_labels[:, column])

        # Calculate the distribution for each value in the column
        distributions[:, column] = np.round(matrix_predicted_labels[:, column] / column_sum * 100, 2)

    headers = list(action_labels_dic.values())
    rcolors = plt.cm.BuPu(np.full(len(headers), 0.1))
    the_table = plt.table(cellText=distributions,
                          rowLabels=headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=rcolors,
                          colLabels=headers,
                          cellLoc='center',
                          loc='center')

    plt.axis('off')
    plt.title("Predicted actions in Swap metric")
    plt.show()
    # plt.savefig('table_image.png')  # Save the image as a file
    plt.close()  # Close the plot to release resources

    if run:
        if type == "train":
            log_to_neptune(run, {type + '/swap/acc_motion': np.mean(values_motion_swap),
                                 type + '/swap/acc_static': np.mean(values_static_swap)})
        else:
            log_to_neptune(run, {'swap/acc_motion': np.mean(values_motion_swap),
                                 'swap/acc_static': np.mean(values_static_swap)})

    return np.mean(values_motion_swap), np.mean(values_static_swap)


def check_cls_mug_eval(opt, model, classifier, test_loader, type_gt):
    e_values_action, e_values_subj = [], []
    mistake_sequences = []
    flag = 0
    for epoch in range(opt.niter):
        print("Epoch", epoch)
        model.eval()
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        torch.cuda.empty_cache()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = data[0].cuda(), data[1].cuda(), data[
                2].cuda()  # x: (128, 15, 3, 64, 64)) label_A: (128,) label_D: (128,)

            if opt.type_gt == "action":
                recon_x_sample, recon_x = model.forward_fixed_action_for_classification(
                    x)  # recon_x_sample: (128, 15, 3, 64, 64) recon_x: (128, 15, 3, 64, 64)
            else:
                recon_x_sample, recon_x = model.forward_fixed_content_for_classification(
                    x)  # recon_x_sample: (128, 15, 3, 64, 64) recon_x: (128, 15, 3, 64, 64)

            with torch.no_grad():
                pred_action1, pred_sub1 = classifier(x)  # pred_action1: (128, 6) pred_sub1: (128, 52)
                pred_action2, pred_sub2 = classifier(recon_x_sample)  # pred_action2: (128, 6) pred_sub2: (128, 52)

                pred1 = F.softmax(pred_action1, dim=1)  # pred1: (128, 6)
                pred2 = F.softmax(pred_action2, dim=1)  # pred2: (128, 6)

            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)  # label2: (128,)
            label2_all.append(label2)  # label2_all{list:1}: (128,)

            # Find mistakes made by the classifier and collect the corresponding sequences
            mistake_indices = np.where(label2 != label_D.cpu().numpy())[0]
            for idx in mistake_indices:
                mistake_sequences.append(x[idx].cpu().numpy())

            pred1_all.append(pred1.detach().cpu().numpy())  # pred1_all{list:1}: (128, 6)
            pred2_all.append(pred2.detach().cpu().numpy())  # pred2_all{list:1}: (128, 6)
            label_gt.append(label_D.detach().cpu().numpy())  # label_gt{list:1}: (128,)

            acc0_sample = count_D(np.argmax(pred_action2.detach().cpu().numpy(), axis=1),
                                  label_A.cpu().numpy()).mean()
            acc1_sample = (np.argmax(pred_sub2.detach().cpu().numpy(), axis=1) == label_D.cpu().numpy()).mean()

            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample
            torch.cuda.empty_cache()
            # Plot the first few mistaken sequences
            if flag < 1:
                for k, seq in enumerate(mistake_sequences[:10]):
                    plt.figure(figsize=(10, 2))
                    for j, frame in enumerate(seq):
                        plt.subplot(1, len(seq), j + 1)
                        plt.imshow(frame.transpose((1, 2, 0)))  # assuming the image data is in CHW format
                        plt.axis('off')
                    plt.title(type_gt)
                    plt.show()
                    flag += 1

        print(
            'Test sample: action_Acc: {:.2f}% subj_Acc: {:.2f}%'.format(
                mean_acc0_sample / len(test_loader) * 100,
                mean_acc1_sample / len(test_loader) * 100))

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
        e_values_subj.append(mean_acc1_sample / len(test_loader) * 100)

    return np.mean(e_values_action), np.mean(e_values_subj), mistake_sequences



def check_cls_mug_table(opt, model, classifier, test_loader, run, type='test'):
    e_values_action, e_values_subj = [], []
    matrix_predicted_labels = np.zeros((6, 6))
    for epoch in range(opt.niter):
        print("Epoch", epoch)
        model.eval()
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = data[0].cuda(), data[1].cuda(), data[
                2].cuda()  # x: (batch_size, 15, 3, 64, 64) label_A: (batch_size), label_D: (batch_size)

            if opt.type_gt == "action":
                recon_x_sample, recon_x = model.forward_fixed_action_for_classification(
                    x)  # recon_x_sample: (batch_size, 15, 3, 64, 64), recon_x: (batch_size, 15, 3, 64, 64)
            else:
                recon_x_sample, recon_x = model.forward_fixed_content_for_classification(x)

            with torch.no_grad():
                pred_action1, pred_sub1 = classifier(x)  # pred_action1: (batch_size, 6), pred_sub1: (batch_size, 52)
                pred_action2, pred_sub2 = classifier(
                    recon_x_sample)  # pred_action2: (batch_size, 6), pred_sub2: (batch_size, 52)

                pred1 = F.softmax(pred_action1, dim=1)  # pred1: (batch_size, 6)
                pred2 = F.softmax(pred_action2, dim=1)  # pred2: (batch_size, 6)

            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)  # label2: (batch_size)
            label2_all.append(label2)

            pred1_all.append(pred1.detach().cpu().numpy())
            pred2_all.append(pred2.detach().cpu().numpy())
            label_gt.append(label_D.detach().cpu().numpy())

            if opt.type_gt == "action":
                for true_label, predicted_label in zip(label_A, label2):
                    matrix_predicted_labels[predicted_label][true_label] = matrix_predicted_labels[predicted_label][
                                                                               true_label] + 1

            acc0_sample = count_D(np.argmax(pred_action2.detach().cpu().numpy(), axis=1), label_A.cpu().numpy()).mean()
            acc1_sample = (np.argmax(pred_sub2.detach().cpu().numpy(), axis=1) == label_D.cpu().numpy()).mean()

            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample

        print(
            'Test sample: action_Acc: {:.2f}% subj_Acc: {:.2f}%'.format(
                mean_acc0_sample / len(test_loader) * 100,
                mean_acc1_sample / len(test_loader) * 100))

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
        e_values_subj.append(mean_acc1_sample / len(test_loader) * 100)
    if opt.type_gt == "action":
        distributions = np.zeros((6, 6))
        for column in range(6):
            # Get the sum of the column
            column_sum = np.sum(matrix_predicted_labels[:, column])

            # Calculate the distribution for each value in the column
            distributions[:, column] = np.round(matrix_predicted_labels[:, column] / column_sum * 100, 2)
        headers = list(action_labels_dic.values())
        rcolors = plt.cm.BuPu(np.full(len(headers), 0.1))
        the_table = plt.table(cellText=distributions,
                              rowLabels=headers,
                              rowColours=rcolors,
                              rowLoc='right',
                              colColours=rcolors,
                              colLabels=headers,
                              cellLoc='center',
                              loc='center')

        plt.axis('off')
        plt.title("Predicted actions in Check-CLS metric")
        plt.show()
