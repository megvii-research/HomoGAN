import logging
import random

import cv2
import numpy as np
import torch
import torchvision

_logger = logging.getLogger(__name__)


class RandomCrop(object):
    def __init__(self, size=64):
        self.size = size

    def __call__(self, img):
        h, w, _ = img.shape
        th, tw = self.size, self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img[y1:y1 + th, x1:x1 + tw]
        return img


class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = np.copy(np.fliplr(img))
        return img


class ArrayToTensor(object):
    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)
        return tensor.float()


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input


def fetch_transform(params):
    if params.transform_type == "basic":
        train_transforms = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(64),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ]

        test_transforms = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
            torchvision.transforms.ToTensor()
        ]
    elif params.transform_type == "homo":
        train_transforms = [
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.GaussianBlur(7, 2.0),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomErasing(),
            torchvision.transforms.Normalize((0.402, 0.446, 0.466,), (0.284, 0.270, 0.274)),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]
        test_transforms = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(params.patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.402, 0.446, 0.466,), (0.284, 0.270, 0.274)),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]

    _logger.info("Train transforms: {}".format(", ".join([type(t).__name__ for t in train_transforms])))
    _logger.info("Val and Test transforms: {}".format(", ".join([type(t).__name__ for t in test_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    test_transforms = torchvision.transforms.Compose(test_transforms)
    return train_transforms, test_transforms


if __name__ == '__main__':
    print("hello world")
