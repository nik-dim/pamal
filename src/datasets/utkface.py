import logging
from typing import List

import torch
import torchvision.transforms as T
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

from src.datasets.utils.enums import TaskCategoriesEnum
from src.datasets.base_data_module import BaseDataModule

SPLIT = [0.7, 0.1, 0.2]


class UTKFaceDataset(ImageFolder):
    """
    The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

    [age] is an integer from 0 to 116, indicating the age
    [gender] is either 0 (male) or 1 (female)
    [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
    """

    def __init__(self, task_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_ids = task_ids
        ages, genders, races = [], [], []
        for i, sample in enumerate(self.samples):
            sample = sample[0]
            a, g, r = sample.split("/")[-1].split(".")[0].split("_")[:-1]
            ages.append(int(a))
            genders.append(int(g))
            races.append(int(r))

        ages = torch.Tensor(ages)
        mean = ages.mean()
        std = ages.std()
        self.ages = (ages - mean) / std
        self.genders = torch.LongTensor(genders)
        self.races = torch.LongTensor(races)

    def __getitem__(self, index: int):
        x, _ = super().__getitem__(index)

        a = self.ages[index]
        g = self.genders[index]
        r = self.races[index]

        target = (a, g, r)

        if self.task_ids is not None:
            target = tuple(tt for i, tt in enumerate(target) if i in self.task_ids)
        return x, target


class UTKFaceDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        logging.info(kwargs)
        self.task_ids = kwargs.get("task_ids", None)
        logging.info(self.task_ids)
        super().__init__(*args, **kwargs)
        # self.train_collate_fn = lambda x: x

    @property
    def name(self):
        return "UTKFace"

    @property
    def input_dims(self):
        return [3, 64, 64]

    @property
    def num_tasks(self):
        return 3

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.REGRESSION, TaskCategoriesEnum.CLASSIFICATION, TaskCategoriesEnum.CLASSIFICATION]

    @property
    def task_names(self) -> List[str]:
        return ["age", "gender", "race"]

    def prepare_data(self, *args, **kwargs):
        self.root = self.root + "utkface"
        logging.debug(self.root)
        dataset = UTKFaceDataset(
            root=self.root,
            transform=T.Compose(
                [
                    T.Resize((64, 64)),
                    T.ToTensor(),
                ]
            ),
            task_ids=self.task_ids,
        )

        split = SPLIT
        logging.warning(f"Splitting train to train, valid, test {split}")
        len_train = int(len(dataset) * split[0])
        len_valid = int(len(dataset) * split[1])
        len_test = len(dataset) - len_train - len_valid
        # len_train = int(len(dataset) * 0.8)
        # len_valid = len(dataset) - len_train
        self.train, self.valid, self.test = torch.utils.data.random_split(
            dataset,
            [len_train, len_valid, len_test],
            generator=torch.Generator().manual_seed(42),
        )


class UTKFaceDatasetNew(TensorDataset):
    def __init__(self, *args, **kwargs):
        tensors = torch.load("/home/ndimitri/dev/base-pt/src/datasets/data/utkface.pt")
        super().__init__(*tensors.values())

    def __getitem__(self, index: int):
        x, a, g, r = super().__getitem__(index)
        target = (a, g, r)
        return x, target


class UTKFaceDataModuleFast(UTKFaceDataModule):
    def prepare_data(self, *args, **kwargs):
        dataset = UTKFaceDatasetNew()

        split = SPLIT
        logging.warning(f"Splitting train to train, valid, test {split}")
        len_train = int(len(dataset) * split[0])
        len_valid = int(len(dataset) * split[1])
        len_test = len(dataset) - len_train - len_valid
        self.train, self.valid, self.test = torch.utils.data.random_split(
            dataset,
            [len_train, len_valid, len_test],
            generator=torch.Generator().manual_seed(42),
        )
