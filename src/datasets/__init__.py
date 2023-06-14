from src.datasets.base_data_module import BaseDataModule
from src.datasets.celeba import CelebaDataModule
from src.datasets.census import CensusDataModule, LinCensusDataModule, MaCensusDataModule
from src.datasets.multimnist import MultiMnistDataModule
from src.datasets.multimnist3digits import MultiMnistThreeDataModule
from src.datasets.utkface import UTKFaceDataModule

__all__ = [
    "BaseDataModule",
    "CelebaDataModule",
    "CensusDataModule",
    "LinCensusDataModule",
    "MaCensusDataModule",
    "Cifar10DataModule",
    "MultiMnistDataModule",
    "MultiMnistThreeDataModule",
    "MultiFashionDataModule",
    "UTKFaceDataModule",
]
