import logging
import os
import random
from argparse import Namespace
from typing import List, Optional

import numpy as np
import torch

from src.datasets.utils.enums import TaskCategoriesEnum


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseDataModule:
    """Base class for all data modules. This class is used to define the data module interface. The design is similar
    to PyTorch Lightning's LightningDataModule."""

    def __init__(
        self,
        root: str = os.path.join(os.path.expanduser("~"), "benchmarks/data/"),
        batch_size: int = 512,
        valid_batch_size: Optional[int] = None,
        num_workers: int = 30,
        shuffle: bool = True,
        pin_memory: bool = True,
        # seed: int = 42,
    ):
        """Inits the BaseDataModule class.

        Args:
            root (str, optional): The directory storing the dataset files. Defaults
                to os.path.join(os.path.expanduser("~"), "benchmarks/data/").
            batch_size (int, optional): The training batch size. Defaults to 512.
            valid_batch_size (Optional[int], optional): The batch size used during evaluation. If not specified, it is
                set to the training batch size. Defaults to None.
            num_workers (int, optional): The number of workers for the PyTorch Dataloader. Defaults to 30.
            shuffle (bool, optional): The shuffle argument for the PyTorch Dataloader. Defaults to True.
            pin_memory (bool, optional): The pin_memory argument for the PyTorch Dataloader. Defaults to True.
        """
        # parse arguments to class
        self.root = root
        self.batch_size = batch_size
        if not isinstance(valid_batch_size, int):
            self.valid_batch_size = batch_size
        else:
            self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        # self.seed = seed
        self.prepare_data()

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def num_tasks(self) -> int:
        raise NotImplementedError

    @property
    def task_categories(self) -> TaskCategoriesEnum:
        raise NotImplementedError

    @property
    def task_names(self) -> List[str]:
        raise NotImplementedError

    def prepare_data(self) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # g = torch.Generator()
        # g.manual_seed(0)

        dl = torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            # generator=g,
        )
        logging.info(f"TRAIN_DL: batch_size={self.batch_size}, num_workers={self.num_workers}")
        return dl

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.valid,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def fetch_train_dataset(self):
        return self.train

    def fetch_valid_dataset(self):
        return self.valid

    def fetch_test_dataset(self):
        return self.test

    def fetch_dataset_args(self):
        return Namespace(
            name=self.name, num_tasks=self.num_tasks, task_categories=self.task_categories, task_names=self.task_names
        )

    @staticmethod
    def __download__(root):
        raise NotImplementedError
