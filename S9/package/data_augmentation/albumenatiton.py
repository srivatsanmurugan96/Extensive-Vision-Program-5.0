from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np


class AlbumentationTransform(object):

    def __init__(self, train=1):
        if (train == 1):
            self.transforms = Compose([
                # transforms.RandomCrop(32, padding=4
                HorizontalFlip(p=.5),
                Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
                Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),  # RGB mean and variance
                ToTensor(),
            ])
        else:
            self.transforms = Compose([
                Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                ToTensor(),
            ])

    def __call__(self, img):

        img = np.array(img)
        img = self.transforms(image=img)["image"]

        return img