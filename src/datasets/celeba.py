import logging
import os
from pathlib import Path
from typing import Any, Tuple

import torchvision.transforms as transforms
from src.datasets.base_data_module import BaseDataModule
from src.datasets.utils.enums import TaskCategoriesEnum
from torchvision.datasets import CelebA
from torchvision.datasets.utils import download_file_from_google_drive


class CelebaDataModule(BaseDataModule):
    """The data module for Celeb-A. Inherits from BaseDataModule. The dataset consists of 10,177 people, 202,599 images and 40 binary classification tasks.

    See more at https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
    """

    def __init__(
        self,
        root=None,
        transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]),
        *args,
        **kwargs,
    ):
        """Inits the data module for the celeba dataset. Inherits from BaseDataModule. See more at the parent class.

        Args:
            transform (callable, optional): transform to be applied to the dataset. Defaults to transforms.ToTensor().
        """
        print(f"kwargs: {kwargs}")
        self.task_ids = kwargs.get("task_ids", None)
        print(self.task_ids)
        if root is None:
            root = Path.joinpath(Path.home(), "data")
        self.transform = transform
        super().__init__(root=root, *args, **kwargs)

    @property
    def name(self):
        return "celeba"

    @property
    def num_tasks(self):
        return 40

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.BINARY_CLASSIFICATION] * self.num_tasks

    def prepare_data(self, *args, **kwargs):
        """Prepares the data for Celeb-A dataset. The dataset comes with predefined train/val/test splits."""

        self.train = CustomCelebA(
            root=self.root,
            split="train",
            download=True,
            transform=self.transform,
            task_ids=self.task_ids,
        )

        self.valid = CustomCelebA(
            root=self.root,
            split="valid",
            download=True,
            transform=self.transform,
            task_ids=self.task_ids,
        )

        self.test = CustomCelebA(
            root=self.root,
            split="test",
            download=True,
            transform=self.transform,
            task_ids=self.task_ids,
        )


class CustomCelebA(CelebA):
    """Wrapper class for CelebA. Essentially it replaces the google drive downloading from the default location to
    my personal drive. This is due to the quotas imposed by Google drive resulting in inability to download from the
    default location."""

    def __init__(self, *args, **kwargs) -> None:
        self.task_ids = kwargs.pop("task_ids", None)
        super().__init__(*args, **kwargs)
        if self.task_ids is not None:
            logging.info(f"using task ids: {self.task_ids}")
            logging.info([a for i, a in enumerate(self.attr_names) if i in self.task_ids])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, target = super().__getitem__(index)
        target = tuple(tt for tt in target)

        if self.task_ids is not None:
            target = tuple(tt for i, tt in enumerate(target) if i in self.task_ids)

        return X, target

    def download(self) -> None:
        pass
        # self.__class__.download(self.root)

    def _check_integrity(self) -> bool:
        return True

    @staticmethod
    def _download(root):
        """Downloads the dataset from my personal Google drive.

        Args:
            root (str): where to store the downloaded files.
        """
        path = Path.joinpath(Path().resolve(), Path(root), "data")
        path = Path.joinpath(path, "celeba")

        if not path.exists():
            print(f"creating dirs for {str(path)}")
            path.mkdir(parents=True, exist_ok=True)

        base_url = "https://drive.google.com/uc?id="
        ids = {
            "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS": "identity_celebA.txt",
            "1BcfJ3AN4NK_E7to3-ZRO_zSCHMUa3EhQ": "list_attr_celeba.txt",
            "1dhtLE_2v2ELQ0ItUQpU-EoIWyJ91t4OI": "list_bbox_celeba.txt",
            "1yVG9OPmOv8jaAbfy4NZdKmM290K026xm": "list_landmarks_align_celeba.txt",
            "13r-ohk_4QfQHA4eQjdeGAJVhrmQD3ubr": "list_landmarks_celeba.txt",
            "1hzynUBVUQWgpEAZXsp3JKflwlEv0p5Ys": "list_eval_partition.txt",
            "19yhnQFuJgnlPQXsYT3TarPLv67agtvF1": "img_align_celeba.zip",
        }

        for file_id, name in ids.items():
            file_output = Path.joinpath(path, name)

            if not os.path.exists(file_output):
                file_url = "%s%s" % (base_url, file_id)
                download_file_from_google_drive(file_id=file_id, root=path, filename=name)
