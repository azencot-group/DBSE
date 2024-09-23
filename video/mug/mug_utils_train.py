import torch
import torch.nn.functional as F
import numpy as np
from utils import KL_divergence, inception_score, entropy_Hy, entropy_Hyx
import matplotlib.pyplot as plt


action_labels_dic = {
    0: 'anger',
    1: 'disgust',
    2: 'happiness',
    3: 'fear',
    4: 'sadness',
    5: 'surprise',
}


def count_D(pred, label, mode=1):
    """Check if predictions match the labels based on the specified mode.

    :param pred: Predicted values.
    :param label: Ground truth labels.
    :param mode: Mode for comparison (default is 1).
    :return: Boolean array indicating whether predictions match labels.
    """
    return (pred // mode) == (label // mode)


def check_cls_mug(opt, model, classifier, test_loader, run):
    """Evaluate the model's classification accuracy on the MUG dataset.

    :param opt: Options/parameters for evaluation.
    :param model: The model to evaluate.
    :param classifier: The classifier to use for predictions.
    :param test_loader: DataLoader for the test dataset.
    :return: Average accuracy for action and subject classifications.
    """
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

    return np.mean(e_values_action), np.mean(e_values_subj)


def pair_iterable(iterable):
    """Create pairs from an iterable.

    :param iterable: Iterable to create pairs from.
    :return: Iterator of paired elements.
    """
    """Create pairs from iterable."""
    iterable = iter(iterable)
    return zip(iterable, iterable)


def test_swap(opt, model, classifier, test_loader, run, type='test'):
    """Test the model's ability to swap motion and static content.

    :param opt: Options for the test.
    :param model: The model to evaluate.
    :param classifier: The classifier to use for predictions.
    :param test_loader: DataLoader for the test dataset.
    :param run: Identifier for the run.
    :param type: Type of test (default is 'test').
    :return: Average accuracy for motion and static content swaps.
    """
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


def arg_max(pred):
    """Get the index of the maximum value in the predictions.

    :param pred: Predicted values.
    :return: Indices of the maximum values.
    """
    return np.argmax(pred.detach().cpu().numpy(), axis=1)


def get_labels(classifier, data):
    """Obtain action and subject labels from the classifier.

    :param classifier: The classifier to use for predictions.
    :param data: Input data for prediction.
    :return: Tuple of action and subject labels.
    """
    with torch.no_grad():
        pred_action, pred_subject = classifier(data)
    return arg_max(pred_action), arg_max(pred_subject)


def create_dict(name, orig_action, pred_action, orig_subject, pred_subject):
    """Create a dictionary to store image predictions and labels.

    :param name: Image name.
    :param orig_action: Original action label.
    :param pred_action: Predicted action label.
    :param orig_subject: Original subject label.
    :param pred_subject: Predicted subject label.
    :return: Dictionary containing image name and labels.
    """
    return {'Img': name,
            'A': action_labels_dic[orig_action],
            'Pred A': action_labels_dic[pred_action],
            'S': orig_subject,
            'Pred S': pred_subject,
            'A_correct?': pred_action == orig_action,
            'S_correct?': pred_subject == orig_subject}


def test_swap_table(opt, model, classifier, test_loader, run, type='test'):
    """Evaluate and visualize swap actions in a table format.

    :param opt: Options for the test.
    :param model: The model to evaluate.
    :param classifier: The classifier to use for predictions.
    :param test_loader: DataLoader for the test dataset.
    :param run: Identifier for the run.
    :param type: Type of test (default is 'test').
    :return: Average accuracy for motion and static content swaps.
    """
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

    return np.mean(values_motion_swap), np.mean(values_static_swap)


def check_cls_mug_eval(opt, model, classifier, test_loader, type_gt):
    """Evaluate classification accuracy and collect mistake sequences.

    :param opt: Options/parameters for evaluation.
    :param model: The model to evaluate.
    :param classifier: The classifier to use for predictions.
    :param test_loader: DataLoader for the test dataset.
    :param type_gt: Type of ground truth (action or subject).
    :return: Average accuracy for action and subject classifications, and mistake sequences.
    """
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
    """Evaluate classification accuracy and visualize results in a table format.

    :param opt: Options/parameters for evaluation.
    :param model: The model to evaluate.
    :param classifier: The classifier to use for predictions.
    :param test_loader: DataLoader for the test dataset.
    :param run: Identifier for the run.
    :param type: Type of test (default is 'test').
    :return: Average accuracy for action and subject classifications.
    """
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
