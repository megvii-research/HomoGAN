"""Train the model"""

import argparse
import datetime
import os
import itertools
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.losses import compute_losses, Mask_Loss, compute_gradient_penalty
from model.GradientReversal.functional import revgrad

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/pre_train/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")
parser.add_argument('-ow', '--only_weights', action='store_true',
                    help='Only use weights to load or load all train status.')
parser.add_argument('--seed', type=int, default=230, help='random seed')


def train(model, manager):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # loss status and val/test status initial
    manager.reset_loss_status()
    HNet, DNet = model

    # set model to training mode
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)

    HNet.train()
    if not params.pretrain_phase:
        DNet.train()
    mk_loss = Mask_Loss()

    with tqdm(total=len(manager.dataloaders['train']), ncols=100) as t:
        for i, data_batch in enumerate(manager.dataloaders['train']):
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            # infor print
            print_str = manager.print_train_info()

            # compute model output and loss
            output = HNet(data_batch)

            if params.pretrain_phase:
                loss = {}
                loss.update(compute_losses(data_batch, output, manager.params))
                loss['total'] = params.h_weight * loss['h_total']

            else:
                # ==================================mask===================================================
                img1_patch_mask, img2_patch_mask = output['img1_patch_mask'], output['img2_patch_mask']

                # =============================feature pair ===================================================
                """
                detach feature to avoid the adversarial loss affecting the feature extractor.
                """

                img1_patch_fea, img2_patch_fea = output["img1_patch_fea"].detach(), output["img2_patch_fea"].detach()
                warp_img1_patch_fea, warp_img2_patch_fea = output["warp_img1_patch_fea"].detach(), output["warp_img2_patch_fea"].detach()

                mask_pair = torch.cat([img1_patch_fea * img1_patch_mask, img2_patch_fea * img2_patch_mask], dim=1)
                warp_pair1 = torch.cat([img1_patch_fea, warp_img1_patch_fea], dim=1)
                warp_pair2 = torch.cat([img2_patch_fea, warp_img2_patch_fea], dim=1)

                # ==========================Gradient_Reversal_Layer=================================================
                if params.dynamic_apha:
                    len_dataloader = len(manager.dataloaders['train'])
                    p = float(i + manager.epoch * len_dataloader) / params.num_epochs / len_dataloader

                    alpha = max(2. / (1. + np.exp(-10 * p)) - 1, 0.1)
                else:
                    alpha = 0.5
                mask_pair = revgrad(mask_pair, alpha)

                gradient_penalty = (compute_gradient_penalty(DNet, mask_pair.detach(), warp_pair1.detach()) \
                                    + compute_gradient_penalty(DNet, mask_pair.detach(), warp_pair2.detach())) / 2
                loss = {'mp_loss': torch.mean(DNet(mask_pair)),
                        'wp_loss': (torch.mean(DNet(warp_pair1.detach())) + torch.mean(DNet(warp_pair2.detach()))) / 2,
                        'mask_value': (torch.mean(img1_patch_mask) + torch.mean(img2_patch_mask)) / 2,
                        'grad': gradient_penalty,
                        'mask_loss': (mk_loss(img1_patch_mask) + mk_loss(img2_patch_mask)) / 2
                        }

                # ===========================H phase=================================================
                loss.update(compute_losses(data_batch, output, manager.params))
                loss['total'] = params.h_weight * loss['h_total'] + params.cls_weight * (
                        -loss['wp_loss'] + loss['mp_loss']) + 10 * gradient_penalty + params.mk_weight * loss['mask_loss']
            # update loss status and print current loss and average loss
            manager.update_loss_status(loss=loss, split="train")
            manager.optimizer.zero_grad()
            loss['total'].backward()
            manager.optimizer.step()
            manager.update_step()

            if i % 2000 == 0 and i != 0:
                print('\n')
                val_metrics = evaluate(model, manager)
                avg = val_metrics['MSE_avg']
                #manager.writer.add_scalar('Evaluate_Results', avg, manager.step)

                manager.cur_val_score = avg
                manager.check_best_save_last_checkpoints(latest_freq=1)

            t.set_description(desc=print_str)
            t.update()

    manager.scheduler.step()
    manager.update_epoch()


def train_and_evaluate(model, manager):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
    """

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()

    for epoch in range(manager.params.num_epochs):
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)

        # Evaluate for one epoch on validation set
        evaluate(model, manager)

        # Save best model weights accroding to the params.major_metric
        manager.check_best_save_last_checkpoints(latest_freq=1)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Update args into params
    params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    if params.cuda:
        torch.cuda.manual_seed(args.seed)

    logger = utils.set_logger(os.path.join(params.model_dir, 'train.log'))
  
    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the input data pipeline
    logger.info("Loading the datasets from {}".format(params.data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.pretrain_phase:
        if params.cuda:
            HNet= net.fetch_net(params).cuda()
            HNet = torch.nn.DataParallel(HNet, device_ids=range(torch.cuda.device_count()))
        else:
            HNet = net.fetch_net(params)

        optimizer = optim.Adam(HNet.parameters(), lr=params.learning_rate)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

        # initial status for checkpoint manager
        manager = Manager(model=[HNet, None], optimizer=optimizer,
                          scheduler=scheduler, params=params, dataloaders=dataloaders,
                          writer=None, logger=logger)

        # Train the model
        logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

        train_and_evaluate([HNet, None], manager)

    else:
        if params.cuda:
            HNet, DNet = list(map(lambda x: x.cuda(), net.fetch_net(params)))
            HNet = torch.nn.DataParallel(HNet, device_ids=range(torch.cuda.device_count()))
            DNet = torch.nn.DataParallel(DNet, device_ids=range(torch.cuda.device_count()))
        else:
            HNet, DNet = net.fetch_net(params)

        optimizer = optim.Adam(itertools.chain(HNet.parameters(), DNet.parameters()), lr=params.learning_rate)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

        # initial status for checkpoint manager
        manager = Manager(model=[HNet, DNet], optimizer=optimizer,
                          scheduler=scheduler, params=params, dataloaders=dataloaders,
                          writer=None, logger=logger)

        # Train the model
        logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

        train_and_evaluate([HNet, DNet], manager)
