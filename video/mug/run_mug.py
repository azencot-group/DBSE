import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import progressbar
import torch.optim as optim
import torch.utils.data
from video.model.dbse_model import DBSE
from video.utils.video_utils import *
from video.mug.mug_utils_train import *
from mug_hyperparameters import *
from video.model.dbse_utils import DbseLoss
from mug_utils import *

mse_loss = nn.MSELoss().cuda()


def pre_train(opt):
    """
    Prepares and initializes the training components including model, optimizer, scheduler, and data loaders.

    :param opt: Options or configuration settings for training.
    :return: Classifier, epoch loss, model, optimizer, run, scheduler, test loader, train loader.
    """
    run = None
    opt.rng = 1234

    define_seed(opt)

    opt.optimizer = optim.Adam
    model = DBSE(opt)
    model.apply(init_weights)
    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    scheduler = define_scheduler(opt, optimizer)

    model = model.cuda()

    classifier = define_classifier(opt)

    test_loader, train_loader = load_dataset(opt)

    epoch_loss = DbseLoss()

    opt.rng = np.random.default_rng(1234)

    return classifier, epoch_loss, model, optimizer, run, scheduler, test_loader, train_loader


def process_train(classifier, epoch_loss, model, opt, optimizer, run, scheduler, test_loader, train_loader):
    """
    Processes the training loop for multiple epochs.

    :param classifier: The classifier model.
    :param epoch_loss: The loss object for the epoch.
    :param model: The model to be trained.
    :param opt: Options or configuration settings for training.
    :param optimizer: The optimizer for model training.
    :param run: A logging object.
    :param scheduler: The learning rate scheduler.
    :param test_loader: Data loader for testing.
    :param train_loader: Data loader for training.
    :return: Metrics for action and subject, and a flag indicating whether training should continue.
    """
    beta_np_inc = frange_cycle_cosine(0.0, 1., opt.nEpoch, opt.nEpoch // 100)
    a_action = s_subj = a_subj = s_action = 0
    for epoch in range(1, opt.nEpoch + 1):
        if epoch and scheduler is not None:
            scheduler.step()

        model.train()
        epoch_loss.reset()
        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(maxval=len(train_loader)).start()

        opt.weight_f = beta_np_inc[epoch - 1] * opt.weight_f
        opt.weight_z = beta_np_inc[epoch - 1]

        for i, data in enumerate(train_loader):
            progress.update(i + 1)
            x = data[0].cuda()
            recon_seq, recon_frame, kld_f, kld_z, orthogonal_loss, z_post_first_frame_norm, z_post_mean_first_frame_norm = epoch_train_and_test(
                x, model, optimizer, opt, epoch_num=epoch)

            epoch_loss.update(recon_seq, recon_frame, kld_f, kld_z)

        progress.finish()
        clear_progressbar()

        a_action, s_subj, a_subj, s_action, should_continue = report_test_result_during_train(classifier, epoch, model,
                                                                                              opt, optimizer, run,
                                                                                              test_loader, train_loader)
        if not should_continue:
            return a_action, s_subj, a_subj, s_action, False

    return a_action, s_subj, a_subj, s_action, True


def epoch_train_and_test(x, model, optimizer, opt, mode="train", epoch_num=opt.nEpoch):
    """
    Performs a single training or validation step on the model.

    :param x: Input data for the model.
    :param model: The model to be trained or validated.
    :param optimizer: The optimizer used for training.
    :param opt: Options or configuration settings.
    :param mode: Mode of operation, either 'train' or 'val'.
    :param epoch_num: The current epoch number.
    :return: List of loss metrics: [reconstruction loss, frame loss, KLD loss, orthogonal loss, norms].
    """
    if mode == "train":
        model.zero_grad()

    batch_size = x.size(0)
    f_mean, f_logvar, f_post, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_frame_x, recon_seq_x = model(
        x)

    kld_f, kld_z, l_recon, l_recon_frame, loss, orthogonal_loss = calculate_epoch_loss(batch_size, epoch_num, f_logvar,
                                                                                       f_mean, opt, recon_frame_x,
                                                                                       recon_seq_x, x, z_post_logvar,
                                                                                       z_post_mean, z_prior_logvar,
                                                                                       z_prior_mean, f_post, z_post)
    z_post_first_frame_norm, z_post_mean_first_frame_norm = torch.norm(z_post[:, 0, :]), torch.norm(
        z_post_mean[:, 0, :])

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return [i.data.cpu().numpy() for i in
            [l_recon, l_recon_frame, kld_f, kld_z, orthogonal_loss, z_post_first_frame_norm,
             z_post_mean_first_frame_norm]]


def report_test_result_during_train(classifier, epoch, model, opt, optimizer, run, test_loader, train_loader):
    """
    Reports test results during training and saves the model if performance improves.

    :param classifier: The classifier model.
    :param epoch: Current epoch number.
    :param model: The model being trained.
    :param opt: Options or configuration settings.
    :param optimizer: The optimizer used for training.
    :param run: A logging object.
    :param test_loader: Data loader for testing.
    :param train_loader: Data loader for training.
    :return: Metrics for action and subject, and a flag indicating whether to continue training.
    """
    global best_acc
    a_action = s_subj = a_subj = s_action = 0
    if epoch % opt.evl_interval == 0 or epoch == opt.nEpoch:
        model.eval()

        val_mse_seq = val_mse_frame = val_kld_f = val_kld_z = val_orthogonal_loss = 0.
        for i, data in enumerate(test_loader):
            x = data[0].cuda()

            with torch.no_grad():
                recon_seq, recon_frame, kld_f, kld_z, orthogonal_loss, z_post_first_frame_norm, z_post_mean_first_frame_norm = epoch_train_and_test(
                    x, model, optimizer, opt, mode="val", epoch_num=epoch)

            val_mse_seq += recon_seq
            val_mse_frame += recon_frame
            val_kld_f += kld_f
            val_kld_z += kld_z
            val_orthogonal_loss += orthogonal_loss

        n_batch = len(test_loader)

        opt.type_gt = 'action'
        a_action, a_subj = check_cls_mug(opt, model, classifier, test_loader, run)
        opt.type_gt = 'aaction'
        s_action, s_subj = check_cls_mug(opt, model, classifier, test_loader, run)
        if epoch >= 400 and (a_action < 50 or s_subj < 50 or a_subj > 60 or s_action > 60):
            return a_action, s_subj, a_subj, s_action, False
        if epoch >= 600 and (a_action < 80 or s_subj < 70 or a_subj > 50 or s_action > 50):
            return a_action, s_subj, a_subj, s_action, False

        # test_swap(opt, model, classifier, test_loader, run)

        try:
            if a_action > best_acc:
                best_acc = a_action
                results = 'a_action=' + str(a_action) + 'a_subj=' + str(a_subj) + 's_action=' + str(
                    s_action) + 's_subj=' + str(s_subj)
                net2save = model.module if torch.cuda.device_count() > 1 else model
                torch.save({
                    'model': net2save.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    '%s/model%d%s.pth' % (opt.log_dir, opt.nEpoch, results))
        except Exception as e:
            print(e)

    return a_action, s_subj, a_subj, s_action, True


def calculate_orthogonal_loss(f_post, z_post):
    """
    Calculates the orthogonal loss between the posteriors of two latent variables.

    :param f_post: Posterior of the first latent variable.
    :param z_post: Posterior of the second latent variable.
    :return: Computed orthogonal loss.
    """
    f_expanded = f_post.expand_as(z_post)
    cos_sim = F.cosine_similarity(z_post, f_expanded, dim=-1)
    orthogonal_loss = torch.mean(torch.square(cos_sim))
    return orthogonal_loss


def calculate_epoch_loss(batch_size, epoch_num, f_logvar, f_mean, opt, recon_frame_x, recon_seq_x, x, z_post_logvar,
                         z_post_mean, z_prior_logvar, z_prior_mean, f_post, z_post):
    """
    Calculates the losses for the current epoch.

    :param batch_size: Number of samples in the batch.
    :param epoch_num: Current epoch number.
    :param f_logvar: Log variance of the first latent variable.
    :param f_mean: Mean of the first latent variable.
    :param opt: Options or configuration settings.
    :param recon_frame_x: Reconstructed frame data.
    :param recon_seq_x: Reconstructed sequence data.
    :param x: Original input data.
    :param z_post_logvar: Log variance of the posterior latent variable.
    :param z_post_mean: Mean of the posterior latent variable.
    :param z_prior_logvar: Log variance of the prior latent variable.
    :param z_prior_mean: Mean of the prior latent variable.
    :param f_post: The posterior of the first latent variable.
    :param z_post: The posterior of the second latent variable.
    :return: Loss metrics including reconstruction losses and KLD losses.
    """
    if opt.loss_recon == 'L2':  # True branch
        l_recon = F.mse_loss(recon_seq_x[:, 1:], x[:, 1:], reduction='sum')
    else:
        l_recon = torch.abs(recon_seq_x - x).sum()

    if opt.loss_recon == 'L2':  # True branch
        l_recon_frame = F.mse_loss(recon_frame_x, x[:, 0], reduction='sum')
    else:
        l_recon_frame = torch.abs(recon_frame_x - x[:, 0]).sum()

    f_mean = f_mean.view((-1, f_mean.shape[-1]))  # [128, 256]
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar)  # [128, 8, 32]
    z_prior_var = torch.exp(z_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(
        z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    l_recon, l_recon_frame, kld_f, kld_z = l_recon / batch_size, l_recon_frame / batch_size, kld_f / batch_size, kld_z / batch_size
    orthogonal_loss = calculate_orthogonal_loss(f_post, z_post)

    loss = l_recon * opt.weight_rec_seq + l_recon_frame * opt.weight_rec_frame + kld_f * opt.weight_f + kld_z * opt.weight_z
    return kld_f, kld_z, l_recon, l_recon_frame, loss, orthogonal_loss


def post_train(should_continue):
    """
    Handles the final output after training, indicating whether the training completed successfully
    or if it was stopped early.

    :param should_continue: A boolean flag indicating whether training should continue.
    """
    if should_continue:
        print("Training is complete")
    else:
        print("Early stop has been made")


def main(opt):
    """
    Main function to initiate the training process, including model preparation, training,
    and final reporting.

    :param opt: Options or configuration settings for training.
    :return: Metrics for action and subject classification.
    """
    classifier, epoch_loss, model, optimizer, run, scheduler, test_loader, train_loader = pre_train(opt)
    a_action, s_subj, a_subj, s_action, should_continue = process_train(classifier, epoch_loss, model, opt,
                                                                        optimizer, run, scheduler, test_loader,
                                                                        train_loader)
    post_train(should_continue)

    return a_action, s_subj, a_subj, s_action


if __name__ == '__main__':
    main(opt)
