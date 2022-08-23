import logging
import os
import pickle
import random
import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

from .transformations import fetch_transform

_logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2 ** 32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class HomoValData(Dataset):
    def __init__(self, params, transform, phase):
        assert phase in ["test", "val"]
        self.transform = transform
        self.base_path = params.data_dir
        self.list_path = self.base_path + '{}.txt'.format(
            phase)

        self.data_infor = open(self.list_path, 'r').readlines()
        self.crop_size = params.crop_size
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.shift = params.shift

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):
        img_names = self.data_infor[idx].replace('\n', '')
        img_names = img_names.split(' ')

        img1 = cv2.imread(self.base_path + 'img/' + img_names[0])
        img2 = cv2.imread(self.base_path + 'img/' + img_names[1])

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)), dim=-1).permute(2, 0, 1).float()
        ori_h, ori_w, _ = img1.shape

        img1_rgb = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rgb = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)  
        img2 = np.mean(img2, axis=2, keepdims=True)

        # if img1.shape[0] != self.crop_size[0] or img1.shape[1] != self.crop_size[1]:
        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1, img2, img1_rs, img2_rs, img1_rgb, img2_rgb = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs, img1_rgb, img2_rgb]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0, 1).float()
        imgs_gray_patch = torch.cat((img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()
        imgs_patch_rgb = torch.cat([img1_rgb, img2_rgb], dim=-1).permute(2, 0, 1).float()

        ori_size = torch.Tensor([ori_w, ori_h]).float()
        Ph, Pw = img1_rs.size()

        pts = torch.Tensor([[0, 0], [Pw - 1, 0], [0, Ph - 1], [Pw - 1, Ph - 1]]).float()
        start = torch.Tensor([0, 0]).reshape(2, 1, 1).float()
        data_dict = {"imgs_patch_rgb": imgs_patch_rgb, "imgs_gray_full": imgs_gray_full, "imgs_full": imgs_full,
                     "imgs_gray_patch": imgs_gray_patch, "ori_size": ori_size,
                     "img_name": img_names[0], "pts": pts, "start": start}
        return data_dict


class HomoTestData(Dataset):
    def __init__(self, params, transform, phase):
        assert phase in ["test", "val"]
        self.transform = transform
        self.base_path = params.data_dir
        self.list_path = self.base_path + '{}.txt'.format(
            phase)

        self.data_infor = open(self.list_path, 'r').readlines()
        self.crop_size = params.crop_size
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.shift = params.shift

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):
        img_names = self.data_infor[idx].replace('\n', '')

        video_names = img_names.split('/')[0]
        img_names = img_names.split(' ')

        pt_names = img_names[0].split('/')[-1] + '_' + img_names[1].split('/')[-1] + '.npy'

        img1 = cv2.imread(self.base_path + 'img/' + img_names[0])
        img2 = cv2.imread(self.base_path + 'img/' + img_names[1])

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)), dim=-1).permute(2, 0, 1).float()
        ori_h, ori_w, _ = img1.shape

        pt_set = np.load(self.base_path + 'pt/' + pt_names, allow_pickle=True)
        pt_set = str(pt_set.item())

        img1_rgb = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rgb = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)  
        img2 = np.mean(img2, axis=2, keepdims=True)


        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1, img2, img1_rs, img2_rs, img1_rgb, img2_rgb = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs, img1_rgb, img2_rgb]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0, 1).float()
        imgs_gray_patch = torch.cat((img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()
        imgs_patch_rgb = torch.cat([img1_rgb, img2_rgb], dim=-1).permute(2, 0, 1).float()

        ori_size = torch.Tensor([ori_w, ori_h]).float()
        Ph, Pw = img1_rs.size()

        pts = torch.Tensor([[0, 0], [Pw - 1, 0], [0, Ph - 1], [Pw - 1, Ph - 1]]).float()
        start = torch.Tensor([0, 0]).reshape(2, 1, 1).float()
        data_dict = {"imgs_patch_rgb": imgs_patch_rgb, "imgs_gray_full": imgs_gray_full, "imgs_full": imgs_full,
                     "imgs_gray_patch": imgs_gray_patch, "ori_size": ori_size,
                     "pt_set": pt_set, "video_names": video_names, 'pt_names': pt_names, "pts": pts, "start": start}
        return data_dict


class UnHomoTrainData(Dataset):

    def __init__(self, params, transform, phase='train'):
        assert phase in ['train', 'val', 'test']

        self.patch_size = params.crop_size
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.base_path = params.data_dir
        self.rho = params.rho
        self.normalize = True
        self.horizontal_flip_aug = True
        self.shift = params.shift
        self.transform = transform

        self.list_path = self.base_path + '/{}.txt'.format(
            phase)
        self.data_infor = open(self.list_path, 'r').readlines()

        # others
        self.seed = 0
        random.seed(self.seed)
        random.shuffle(self.data_infor)

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def random_perturb(self, start):
        """
        adding a random warping for fake pair(MaFa, MbFb) and true pair (Fa, Fa'), since there is an interpolation transformation between the original real pair (Fa, Fa')  [easily
         distinguishable by discriminators]
        start: x y

        
        """
        Ph, Pw = self.patch_size

        shift = np.random.randint(-self.shift, self.shift, (4, 2))

        src = np.array([[0, 0], [Pw - 1, 0], [0, Ph - 1], [Pw - 1, Ph - 1]]) + start
        dst = np.copy(src)

        dst[:, 0] = dst[:, 0] + shift[:, 0]
        dst[:, 1] = dst[:, 1] + shift[:, 1]

        H, _ = cv2.findHomography(src, dst)

        return H, shift

    def data_aug(self, img1, img2, horizontal_flip=True, start=None, normalize=True, gray=True):

        def random_crop_tt(img1, img2, start):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.patch_size

            if start is None:
                x = np.random.randint(self.rho,
                                      width - self.rho - patch_size_w)  # [320, 640] --> [patch_size_h, patch_size_w]
                y = np.random.randint(self.rho, height - self.rho - patch_size_h)
                start = [x, y]
            else:
                x, y = start
            img1_patch = img1[y: y + patch_size_h, x: x + patch_size_w, :]
            img2_patch = img2[y: y + patch_size_h, x: x + patch_size_w, :]
            return img1, img2, img1_patch, img2_patch, start

        if horizontal_flip and random.random() <= .5:
            img1 = np.flip(img1, 1)
            img2 = np.flip(img2, 1)

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        img1, img2 = list(map(torch.Tensor, [img1, img2]))

        img1, img2, img1_patch, img2_patch, start = random_crop_tt(img1, img2, start)

        return img1, img2, img1_patch, img2_patch, start

    def __getitem__(self, idx):
        # img loading

        img_names = self.data_infor[idx].replace('\n', '')
        img_names = img_names.split(' ')

        img1 = cv2.imread(self.base_path + 'img/' + img_names[0])
        img2 = cv2.imread(self.base_path + 'img/' + img_names[1])


        # img aug
        img1, img2, img1_patch, img2_patch, start = self.data_aug(img1, img2, self.horizontal_flip_aug)

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0, 1).float()
        imgs_gray_patch = torch.cat((img1_patch, img2_patch), dim=-1).permute(2, 0, 1).float()

        Ph, Pw = self.patch_size
        H_random, _ = self.random_perturb(start=start)
        H_random = torch.Tensor(H_random).float()
        start = torch.Tensor(start).reshape(2, 1, 1).float()
        ori_size = torch.Tensor([Pw, Ph]).float()
        pts = torch.Tensor([[0, 0], [Pw - 1, 0], [0, Ph - 1], [Pw - 1, Ph - 1]]).float()

        # output dict
        data_dict = {"imgs_gray_full": imgs_gray_full, "imgs_gray_patch": imgs_gray_patch, "start": start, "pts": pts,
                     "ori_size": ori_size, "H_random": H_random}

        return data_dict


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    _logger.info("Dataset type: {}, transform type: {}".format(params.dataset_type, params.transform_type))
    train_transforms, test_transforms = fetch_transform(params)

    if params.dataset_type == "homo":
        train_ds = UnHomoTrainData(params, phase='train', transform=train_transforms)
        val_ds = HomoValData(params, phase='val', transform=test_transforms)
        test_ds = HomoTestData(params, phase='test', transform=test_transforms)

    dataloaders = {}
    # add defalt train data loader
    train_dl = DataLoader(
        train_ds,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
        drop_last=True,
        prefetch_factor=3,  # for pytorch >=1.5.0
        worker_init_fn=worker_init_fn)
    dataloaders["train"] = train_dl

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                dl = DataLoader(
                    val_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                    prefetch_factor=3,  # for pytorch >=1.5.0
                )
            elif split == "test":
                dl = DataLoader(
                    test_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                    prefetch_factor=3,  # for pytorch >=1.5.0
                )
            else:
                raise ValueError("Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders
