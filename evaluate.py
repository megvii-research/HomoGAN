"""Evaluates the model"""

import argparse
import logging
import os
import torch.optim as optim
import itertools
import torch
from tqdm import tqdm
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from loss.losses import compute_eval_results
from common.manager import Manager
#from model.utils import save_imgs_mask

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/fine_tuning/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='experiments/fine_tuning/fine_tuning.pth',
                    help="name of the file in --model_dir containing "
                         "weights to load")


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode
    manager.logger.info("eval begin!")

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []
    model, _ = model
    torch.cuda.empty_cache()
    model.eval()
    k = 0
    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("val")

            for data_batch in manager.dataloaders["val"]:
                # move to GPU if available

                # pts_names = data_batch["pt_names"]
                video_name = data_batch["video_names"]

                data_batch = utils.tensor_gpu(data_batch)
                # compute model output
                output = model(data_batch)

                # compute all loss on this batch
                eval_results = compute_eval_results(data_batch, output)
                err_avg = eval_results["errors_m"]

                for j in range(len(err_avg)):
                    k += 1
                    if video_name[j] in RE:
                        MSE_RE.append(err_avg[j])
                    elif video_name[j] in LT:
                        MSE_LT.append(err_avg[j])
                    elif video_name[j] in LL:
                        MSE_LL.append(err_avg[j])
                    elif video_name[j] in SF:
                        MSE_SF.append(err_avg[j])
                    elif video_name[j] in LF:
                        MSE_LF.append(err_avg[j])

            # update data to tensorboard

            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5

            Metric = {"MSE_RE_avg": MSE_RE_avg, "MSE_LT_avg": MSE_LT_avg, "MSE_LL_avg": MSE_LL_avg,
                      "MSE_SF_avg": MSE_SF_avg, "MSE_LF_avg": MSE_LF_avg, "AVG": MSE_avg}
            manager.update_metric_status(metrics=Metric, split="test")

            manager.logger.info(
                "Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                    manager.epoch_val,
                    MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

            # for k, v in manager.val_status.items():
            #     manager.writer.add_scalar("Metric/test/{}".format(k), v.avg, manager.epoch_val)

            # For each epoch, print the metric
            manager.print_metrics("test", title="test", color="green")

            manager.epoch_val += 1

            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)
            model.train()
            val_metrics = {'MSE_avg': MSE_avg}
            return val_metrics

def test(model, manager):
    #Test the model with loading checkpoints.

    # Args:
    #     model: (torch.nn.Module) the neural network
    #     manager: a class instance that contains objects related to train and evaluate.

    # set model to evaluation mode

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    torch.cuda.empty_cache()
    model.eval()
    k = 0
    flag = 0
    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["test"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("test")
            with tqdm(total=len(manager.dataloaders['test']), ncols=100) as t:
                for data_batch in manager.dataloaders["test"]:
                    # move to GPU if available

                    video_name = data_batch["video_names"]
                    data_batch = utils.tensor_gpu(data_batch)
                    # compute model output
                    output_batch = model(data_batch)
                    # compute all loss on this batch

                    flag += 1
                    t.update()
                    eval_results = compute_eval_results(data_batch, output_batch)
                    err_avg = eval_results["errors_m"]
                    for j in range(len(err_avg)):
                        k += 1
                        if video_name[j] in RE:
                            MSE_RE.append(err_avg[j])
                        elif video_name[j] in LT:
                            MSE_LT.append(err_avg[j])
                        elif video_name[j] in LL:
                            MSE_LL.append(err_avg[j])
                        elif video_name[j] in SF:
                            MSE_SF.append(err_avg[j])
                        elif video_name[j] in LF:
                            MSE_LF.append(err_avg[j])
            
                    # update data to tensorboard
                    #print("{}:{}".format(k, err_avg[j]))
            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5
            
            Metric = {"MSE_RE_avg": MSE_RE_avg, "MSE_LT_avg": MSE_LT_avg, "MSE_LL_avg": MSE_LL_avg,
                      "MSE_SF_avg": MSE_SF_avg, "MSE_LF_avg": MSE_LF_avg, "AVG": MSE_avg}
            manager.update_metric_status(metrics=Metric, split="val")
            
            # update data to tensorboard
            
            #manager.writer.add_scalar("Loss/val", MSE_avg, manager.epoch_val)
            manager.logger.info(
                "Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                    manager.epoch_val,
                    MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))
            
            manager.print_metrics("test", title="test", color="red")




if __name__ == '__main__':

    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.pretrain_phase:
        if params.cuda:
            HNet =net.fetch_net(params).cuda()
        else:
            HNet =net.fetch_net(params)
        DNet = None
    else:
        if params.cuda:
            HNet, DNet = list(map(lambda x: x.cuda(), net.fetch_net(params)))
            HNet = torch.nn.DataParallel(HNet, device_ids=range(torch.cuda.device_count()))
            DNet = torch.nn.DataParallel(DNet, device_ids=range(torch.cuda.device_count()))
        else:
            HNet, DNet = net.fetch_net(params)
    if params.pretrain_phase:
        optimizer = optim.Adam(HNet.parameters(), lr=params.learning_rate)
    else:
        optimizer = optim.Adam(itertools.chain(HNet.parameters(), DNet.parameters()), lr=params.learning_rate)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)
    # Initial status for checkpoint manager
    manager = Manager(model=[HNet, DNet], optimizer=optimizer, scheduler=scheduler, params=params,
                      dataloaders=dataloaders,
                      writer=None,
                      logger=logger)

    # Initial status for checkpoint manager

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # test
    test(HNet, manager)

