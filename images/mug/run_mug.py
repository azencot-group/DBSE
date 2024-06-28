import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import progressbar
import torch.optim as optim
import torch.utils.data
from dbse_model import DBSE
from images.mug.utils import *
from images.mug.mug_utils_train import *
from mug_hyperparameters import *
from mug_utils import *
from dbse_loss_mug import DbseLoss
from neptune_utils.neptune_utils import init_neptune, log_to_neptune
# hyperopt dependencies
from hyperopt import fmin, tpe, hp, Trials

### ------------------------------------ Consts ----------------------------------------------- ###
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxODViZjI2Yi05OWFhLTRjZTQtYTE1MS1iNjBmZTJiNzJjYzAifQ=="
NEPTUNE_PROJECT_NAME = "azencot-group/no-mi-new"
NEPTUNE_TAGS = ["MUG", "Experiments", "Frame15"]
MODEL_CONFIG_PARAMS_FOR_RUNNING_NAME = {
    "NA_MUG_epoch": opt.nEpoch,
    "bs": opt.batch_size,
    "decoder": opt.decoder,
    "image_width": opt.image_width,
    "rnn_size": opt.rnn_size,
    "g_dim": opt.g_dim,
    "f_dim": opt.f_dim,
    "z_dim": opt.z_dim,
    "lr": opt.lr,
    "weight:kl_f": opt.weight_f,
    "kl_z": opt.weight_z,
    "rec_seq": opt.weight_rec_seq,
    "rec_frame": opt.weight_rec_frame,
    "loss_recon": opt.loss_recon,
    "sche": opt.sche,
    "note": opt.note
}
best_acc = 0

mse_loss = nn.MSELoss().cuda()


### ------------------------------------ Pre train functions ------------------------------------ ###
def pre_train(opt):
    run = None
    # TODO: check if need
    opt.rng = 1234
    running_name = define_running_name()

    if opt.neptune:
        run = init_neptune(opt, NEPTUNE_PROJECT_NAME, NEPTUNE_API_TOKEN, NEPTUNE_TAGS)

    # # Log running parameters
    # log_file = create_result_and_log_dir(opt, running_name)

    define_seed(opt)

    # print_log('Running parameters:')
    # print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), log_file)

    # Optimizers and model define
    opt.optimizer = optim.Adam
    model = DBSE(opt)
    model.apply(init_weights)
    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    # Define a scheduler to adjust the learning rate during training.
    scheduler = define_scheduler(opt, optimizer)

    # Transfer model to GPU memory (if possible)
    # model = use_multiple_gpus_if_possible(model, log_file, opt)
    model = model.cuda()

    # print_log(model, log_file)

    # Load classifier for testing during train
    classifier = define_classifier(opt)

    # Load a dataset
    test_loader, train_loader = load_dataset(opt)

    # Define loss
    epoch_loss = DbseLoss()

    # Set RNG for permutations
    opt.rng = np.random.default_rng(1234)

    return classifier, epoch_loss, None, model, optimizer, run, scheduler, test_loader, train_loader


def define_running_name():
    return "-".join([f"{k}={v}" for k, v in MODEL_CONFIG_PARAMS_FOR_RUNNING_NAME.items()])


def create_result_and_log_dir(opt, running_name):
    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, running_name)
    log_file = os.path.join(opt.log_dir, 'log.txt')
    summary_dir = os.path.join('./summary/', opt.dataset, running_name)
    # TODO: check if need
    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    # TODO: check if need
    # print_log("Random Seed: {}".format(opt.seed), log_file)
    os.makedirs(summary_dir, exist_ok=True)
    return log_file

    ### ------------------------------------ Process train functions ------------------------------------ ###


def process_train(classifier, epoch_loss, log_file, model, opt, optimizer, run, scheduler, test_loader, train_loader):
    beta_np_inc = frange_cycle_cosine(0.0, 1., opt.nEpoch, opt.nEpoch // 100)
    a_action = s_subj = a_subj = s_action = 0
    for epoch in range(1, opt.nEpoch + 1):
        if epoch and scheduler is not None:
            scheduler.step()

        model.train()
        # Reset loss for log average epoch losses (Logs only)
        epoch_loss.reset()
        opt.epoch_size = len(train_loader)
        # Define console progress bar
        progress = progressbar.ProgressBar(maxval=len(train_loader)).start()

        opt.weight_f = beta_np_inc[epoch - 1] * opt.w_f
        opt.weight_z = beta_np_inc[epoch - 1]

        if opt.neptune:
            run['train/weight_f'].log(opt.weight_f)
            run['train/weight_z'].log(opt.weight_z)

        for i, data in enumerate(train_loader):
            progress.update(i + 1)
            x = data[0].cuda()
            recon_seq, recon_frame, kld_f, kld_z, orthogonal_loss, z_post_first_frame_norm, z_post_mean_first_frame_norm = epoch_train_and_test(
                x, model, optimizer, opt, epoch_num=epoch)

            lr = optimizer.param_groups[0]['lr']
            if opt.neptune:
                log_to_neptune(run, {'train/lr': lr,
                                     'train/mse_seq': recon_seq.item(),
                                     'train/mse_frame': recon_frame.item(),
                                     'train/kld_f': kld_f.item(),
                                     'train/kld_z': kld_z.item(),
                                     'train/orthogonal_loss': orthogonal_loss.item(),
                                     'train/z_post_first_frame_norm': z_post_first_frame_norm.item(),
                                     'train/z_post_mean_first_frame_norm': z_post_mean_first_frame_norm.item()})

            # Update loss for log average epoch losses (Logs only)
            epoch_loss.update(recon_seq, recon_frame, kld_f, kld_z)

        # Clear progress bar when epoch train is finished
        progress.finish()
        clear_progressbar()

        # Log the average loss per epoch (Log only)
        avg_loss = epoch_loss.avg()
        # print_log(f'[{epoch:02d}] recon_seq: {avg_loss[0]:.2f} | recon_frame: {avg_loss[1]:.2f} | kld_f: {avg_loss[2]:.2f} | kld_z: {avg_loss[3]:.2f} | lr: {lr:.5f}', log_file)

        a_action, s_subj, a_subj, s_action, should_continue = report_test_result_during_train(classifier, epoch, model,
                                                                                              opt, optimizer, run,
                                                                                              test_loader, train_loader)
        if not should_continue:
            return a_action, s_subj, a_subj, s_action, False

    return a_action, s_subj, a_subj, s_action, True


def epoch_train_and_test(x, model, optimizer, opt, mode="train", epoch_num=opt.nEpoch):
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

        if opt.neptune:
            log_to_neptune(run, {'test/mse_seq': val_mse_seq.item() / n_batch,
                                 'test/mse_frame': val_mse_frame.item() / n_batch,
                                 'test/kld_f': val_kld_f.item() / n_batch,
                                 'test/kld_z': val_kld_z.item() / n_batch,
                                 'test/orthogonal_loss': val_orthogonal_loss.item() / n_batch,
                                 'train/z_post_first_frame_norm': z_post_first_frame_norm.item(),
                                 'train/z_post_mean_first_frame_norm': z_post_mean_first_frame_norm.item()})
        opt.type_gt = 'action'
        a_action, a_subj = check_cls_mug(opt, model, classifier, test_loader, run)
        opt.type_gt = 'aaction'
        s_action, s_subj = check_cls_mug(opt, model, classifier, test_loader, run)
        if epoch >= 400 and (a_action < 50 or s_subj < 50 or a_subj > 60 or s_action > 60):
            return a_action, s_subj, a_subj, s_action, False
        if epoch >= 600 and (a_action < 80 or s_subj < 70 or a_subj > 50 or s_action > 50):
            return a_action, s_subj, a_subj, s_action, False

        # test_swap(opt, model, classifier, test_loader, run)

        # Save the model
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
    f_expanded = f_post.expand_as(z_post)
    cos_sim = F.cosine_similarity(z_post, f_expanded, dim=-1)
    orthogonal_loss = torch.mean(torch.square(cos_sim))
    return orthogonal_loss


def calculate_epoch_loss(batch_size, epoch_num, f_logvar, f_mean, opt, recon_frame_x, recon_seq_x, x, z_post_logvar,
                         z_post_mean, z_prior_logvar, z_prior_mean, f_post, z_post):
    # Reconstruction for all the frames (both Dynamic and Static)
    if opt.loss_recon == 'L2':  # True branch
        l_recon = F.mse_loss(recon_seq_x[:, 1:], x[:, 1:], reduction='sum')
    else:
        l_recon = torch.abs(recon_seq_x - x).sum()

    # Loss for single using reconstruction with single frame. (Static only)
    if opt.loss_recon == 'L2':  # True branch
        l_recon_frame = F.mse_loss(recon_frame_x, x[:, 0], reduction='sum')
    else:
        l_recon_frame = torch.abs(recon_frame_x - x[:, 0]).sum()

    # KLD Loss compare to normal standard gaussian (Static only)
    f_mean = f_mean.view((-1, f_mean.shape[-1]))  # [128, 256]
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))

    # KLD Loss compare to z prior (Defined in the model) (Dynamic only)
    z_post_var = torch.exp(z_post_logvar)  # [128, 8, 32]
    z_prior_var = torch.exp(z_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(
        z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    # Normalize the losses with the batch size
    l_recon, l_recon_frame, kld_f, kld_z = l_recon / batch_size, l_recon_frame / batch_size, kld_f / batch_size, kld_z / batch_size
    orthogonal_loss = calculate_orthogonal_loss(f_post, z_post)

    # Calculate the complete loss L = Reconstruction_Loss +  Single_Frame_Reconstruction_Loss + KLD_Static + KLD_Dynamic
    loss = l_recon * opt.weight_rec_seq + l_recon_frame * opt.weight_rec_frame + kld_f * opt.weight_f + kld_z * opt.weight_z
    return kld_f, kld_z, l_recon, l_recon_frame, loss, orthogonal_loss

    ### ------------------------------------ Post train functions ------------------------------------ ###


def post_train(opt, run, should_continue):
    if opt.neptune:
        run.stop()
    if should_continue:
        print("Training is complete")
    else:
        print("Early stop has been made")


# Objective function
def objective(hyperparameters):
    # The objective function is what we minimize.
    # We want to maximize a_action, so we minimize negative a_action.
    a_action, s_subj, a_subj, s_action = main(hyperparameters)
    return 100 - a_action + 100 - s_subj + (a_subj + s_action)


def main(hyperparameters):
    # Update opt with hyperparameters
    opt.z_dim = hyperparameters['l_dim']
    opt.f_dim = hyperparameters['l_dim']
    opt.lr = hyperparameters['lr']
    opt.weight_rec_seq = hyperparameters['weight_rec_seq']
    opt.weight_rec_frame = hyperparameters['weight_rec_frame']
    opt.batch_size = hyperparameters['batch_size']
    opt.w_f = hyperparameters['weight_f']

    classifier, epoch_loss, log_file, model, optimizer, run, scheduler, test_loader, train_loader = pre_train(opt)
    a_action, s_subj, a_subj, s_action, should_continue = process_train(classifier, epoch_loss, log_file, model, opt,
                                                                        optimizer, run, scheduler, test_loader,
                                                                        train_loader)
    post_train(opt, run, should_continue)

    return a_action, s_subj, a_subj, s_action


def update_neptune_tags():
    gpu_unique_id = get_gpu_unique_id()
    gpu_ip = get_ip()
    if gpu_ip:
        NEPTUNE_TAGS.append(f"""IP: {gpu_ip}""")
    if gpu_unique_id:
        NEPTUNE_TAGS.append(f"""ID: {gpu_unique_id}""")


def get_experiment_space():
    space = {  # Architecture parameters
        'model': 'dbse',
        'mode': 'simple',
        'l_dim': hp.choice('l_dim', [64]),
        'lr': hp.choice('lr', [0.0015]),
        'weight_rec_seq': hp.choice('weight_rec_seq', [87.]),
        'weight_f': hp.choice('weight_f', [15.]),
        # 'weight_rec_frame': hp.choice('weight_rec_frame', [0.001, .005, 0.01, .1, .2, .5]),
        'weight_rec_frame': hp.choice('weight_rec_frame', [30.]),
        'sche': hp.choice('sche', ['const']),
        # Data parameters
        'batch_size': hp.choice('batch_size', [64])}

    return space


if __name__ == '__main__':
    update_neptune_tags()
    trials = Trials()
    best = fmin(fn=objective, space=get_experiment_space(), algo=tpe.suggest, max_evals=20)
    # main(opt)
