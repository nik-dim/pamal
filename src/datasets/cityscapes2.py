# Adapted from https://github.com/lorenmt/mtan/ and https://github.com/AvivNavon/nash-mtl
import fnmatch
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from src.datasets import BaseDataModule
from torch.utils.data.dataset import Dataset

import copy
from torch.utils.data import random_split


class RandomScaleCropCityScapes(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """

    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(
            img[None, :, i : i + h, j : j + w], size=(height, width), mode="bilinear", align_corners=True
        ).squeeze(0)
        label_ = (
            F.interpolate(label[None, None, i : i + h, j : j + w], size=(height, width), mode="nearest")
            .squeeze(0)
            .squeeze(0)
        )
        depth_ = F.interpolate(depth[None, :, i : i + h, j : j + w], size=(height, width), mode="nearest").squeeze(0)
        return img_, label_, depth_ / sc


class CityScapes(Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """

    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = root
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = Path.joinpath(self.root, "train").as_posix()
        else:
            self.data_path = Path.joinpath(self.root, "val").as_posix()

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + "/image"), "*.npy"))

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + "/image/{:d}.npy".format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + "/label_7/{:d}.npy".format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + "/depth/{:d}.npy".format(index)), -1, 0))

        # apply data augmentation if required
        if self.augmentation:
            image, semantic, depth = RandomScaleCropCityScapes()(image, semantic, depth)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])

        return image.float(), semantic.float(), depth.float()

    def __len__(self):
        return self.data_len


class Cityscapes2DataModule(BaseDataModule):
    num_tasks = 2
    task_names = ["semantic", "depth"]
    name = "Cityscapes2"
    input_dims = [3, 128, 256]

    def __init__(
        self,
        apply_augmentation=True,
        root=Path.joinpath(Path.home(), "benchmarks", "preprocessed", "cityscapes"),
        batch_size=8,
        valid_batch_size=None,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
    ):
        self.apply_augmentation = apply_augmentation
        super().__init__(root, batch_size, valid_batch_size, num_workers, shuffle, pin_memory)
        self.prepare_data()

    def prepare_data(self) -> None:
        self.train = CityScapes(root=self.root, train=True, augmentation=self.apply_augmentation)
        self.valid = CityScapes(root=self.root, train=False)
        self.test = self.valid


class SplitCityScapes(Dataset):
    def __init__(self, root, list_of_indices, train=True, augmentation=False):
        self.train = train
        self.root = root
        self.list_of_indices = list_of_indices
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = Path.joinpath(self.root, "train").as_posix()
        else:
            self.data_path = Path.joinpath(self.root, "val").as_posix()

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + "/image"), "*.npy"))

    def __getitem__(self, index):
        index = self.list_of_indices[index]
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + "/image/{:d}.npy".format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + "/label_7/{:d}.npy".format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + "/depth/{:d}.npy".format(index)), -1, 0))

        # apply data augmentation if required
        if self.augmentation:
            image, semantic, depth = RandomScaleCropCityScapes()(image, semantic, depth)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])

        return image.float(), semantic.float(), depth.float()

    def __len__(self):
        return len(self.list_of_indices)


class Cityscapes2SplitDataModule(BaseDataModule):
    num_tasks = 2
    task_names = ["semantic", "depth"]
    name = "Cityscapes2"
    input_dims = [3, 128, 256]

    def __init__(
        self,
        apply_augmentation=True,
        root=Path.joinpath(Path.home(), "benchmarks", "preprocessed", "cityscapes"),
        batch_size=8,
        valid_batch_size=None,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
    ):
        self.apply_augmentation = apply_augmentation
        super().__init__(root, batch_size, valid_batch_size, num_workers, shuffle, pin_memory)
        self.prepare_data()

    def prepare_data(self) -> None:
        train = CityScapes(root=self.root, train=True, augmentation=self.apply_augmentation)
        train, val = random_split(train, [2500, 475], torch.Generator().manual_seed(42))
        self.train = SplitCityScapes(
            root=train.dataset.root,
            train=True,
            list_of_indices=train.indices,
            augmentation=self.apply_augmentation,
        )

        # do not perform augmentation for validation
        self.valid = SplitCityScapes(
            root=train.dataset.root,
            train=True,
            list_of_indices=val.indices,
            augmentation=False,
        )
        self.test = CityScapes(root=self.root, train=False)
