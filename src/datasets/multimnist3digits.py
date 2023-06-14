import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import resize
from tqdm import tqdm

from src.datasets.base_data_module import BaseDataModule
from src.datasets.utils.enums import TaskCategoriesEnum


class MultiMnistThreeDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return "multimnist3"

    @property
    def input_dims(self):
        return [1, 28, 28]

    @property
    def num_tasks(self):
        return 3

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.CLASSIFICATION] * self.num_tasks

    @property
    def task_names(self) -> List[str]:
        return ["Task 1", "Task 2", "Task 3"]

    def prepare_data(self, *args, **kwargs):
        """Prepares the MultiMNIST dataset."""
        train = MultiMNIST3Digits(
            root=self.root,
            download=True,
            train=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        self.train, self.valid = torch.utils.data.random_split(train, [50000, 10000])

        self.test = MultiMNIST3Digits(
            root=self.root,
            download=True,
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )


class MultiMNIST3Digits(VisionDataset):
    training_file = "training.pt"
    test_file = "test.pt"

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, f"_s={self.s}", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, f"_s={self.s}", "processed")

    def __init__(
        self,
        root: str,
        s=42,
        train: bool = True,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.s = s
        self.train = train
        self._split = "train" if train else "test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not complete." + " You can use download=True to download it")

        # # rgb folder as ground truth
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index].long()
        img = img.view(1, 28, 28)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = tuple(t for t in target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.processed_folder, self.test_file)
        )

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download MNIST
        mnist_train = torchvision.datasets.MNIST(
            root=self.root,
            download=True,
            train=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        mnist_test = torchvision.datasets.MNIST(
            root=self.root,
            download=True,
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        training_set = create_data_samples(mnist_train, num_samples=60000, orig_img_size=self.s)
        test_set = create_data_samples(mnist_test, num_samples=10000, orig_img_size=self.s)

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")


def create_new_sample(ids, dataset, s):
    x0, y0 = dataset[ids[0]]
    x1, y1 = dataset[ids[1]]
    x2, y2 = dataset[ids[2]]

    img = torch.zeros((s, s))

    img[0:28, 0:28] = x0

    a0, a1 = s - 28, s
    b0, b1 = s // 2 - 14, s // 2 + 14
    img[a0:a1, b0:b1] = torch.max(img[a0:a1, b0:b1], x1)

    a0, a1 = 0, 28
    b0, b1 = s - 28, s
    img[a0:a1, b0:b1] = torch.max(img[a0:a1, b0:b1], x2)

    img = resize(img.unsqueeze(0), size=28).squeeze()
    return img, y0, y1, y2


def create_data_samples(dataset, num_samples, orig_img_size):
    """Generates random samples.

    Args:
        dataset (Dataset): the dataset to generate tasks from.
        num_samples (int): the number of samples tp generate.

    Returns:
        tuple: a tuple containing the samples and the targets.
    """
    dataset_length = num_samples
    # dataset_length = 1000

    print(f"Creating {dataset_length} new samples\n")
    custom_data = torch.zeros(dataset_length, 28, 28)
    labels = torch.zeros(dataset_length, 3)
    for i in tqdm(range(dataset_length)):
        ids = np.random.randint(low=0, high=dataset_length, size=(3,))
        while len(set(ids)) < 3:
            ids = np.random.randint(low=0, high=dataset_length, size=(3,))

        img, y0, y1, y2 = create_new_sample(ids, dataset, s=orig_img_size)
        custom_data[i, :, :] = img
        labels[i, :] = torch.LongTensor([y0, y1, y2])

    return (custom_data, labels)
