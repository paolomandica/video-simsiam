import torchvision
import skimage

import torch
from torchvision import transforms

import numpy as np
from PIL import Image

IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD = (0.2023, 0.1994, 0.2010)
NORM = [transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]


class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])

        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack(
                [np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])


def get_frame_transform(img_size, color_aug):

    tt = [
        torchvision.transforms.RandomResizedCrop(
            img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),
        torchvision.transforms.RandomHorizontalFlip(),
    ]

    if color_aug:
        if np.random.rand() <= 0.8:
            tt += [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)]

        tt += [transforms.RandomGrayscale(p=0.2)]

        if np.random.rand() <= 0.5:
            tt += [transforms.GaussianBlur(7)]

    return tt


def get_train_transforms(args):
    norm_size = torchvision.transforms.Resize((args.img_size, args.img_size))
    frame_transform = get_frame_transform(args.img_size, args.color_aug)
    transform = frame_transform + NORM
    train_transform = MapTransform(torchvision.transforms.Compose(transform))
    return train_transform
