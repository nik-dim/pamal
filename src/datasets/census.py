import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

from src.datasets.base_data_module import BaseDataModule
from src.datasets.utils.data_download import _download, unzip
from src.datasets.utils.enums import TaskCategoriesEnum


class MultiTaskTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors: torch.Tensor) -> None:
        super().__init__(*tensors)

    def __getitem__(self, index):
        X, y = super().__getitem__(index)
        y = tuple(yy for yy in y)
        return X, y


class CensusDataModule(BaseDataModule):
    """Base class for DataModule for Census dataset [1] as used in [2].

    References:
        [1] R. Kohavi, “Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid,” in Proceedings of
            the second international conference on knowledge discovery and data mining (KDD-96), 1996.
    """

    num_features: int = 477

    def __init__(
        self,
        income: bool = True,
        age: bool = False,
        education: bool = False,
        never_married: bool = True,
        seed=0,
        *args,
        **kwargs,
    ):
        self.income = income
        self.age = age
        self.education = education
        self.never_married = never_married
        self.seed = seed
        super().__init__(*args, **kwargs)

    @property
    def input_dims(self):
        return [self.num_features]

    @property
    def name(self) -> str:
        return "census"

    @property
    def task_names(self) -> List[str]:
        return ["income", "never_married"]

    @property
    def num_tasks(self):
        return 2

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.BINARY_CLASSIFICATION] * 2

    def prepare_data(self, *args, **kwargs):
        """Prepares the data for the Census dataset."""
        X_train, X_test, y_train, y_test = data_Census_preprocessing(
            self.root,
            income=self.income,
            age=self.age,
            education=self.education,
            never_married=self.never_married,
        )

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)

        tr = transforms.Compose([transforms.ToTensor()])

        self.train = MultiTaskTensorDataset(X_train, y_train)
        self.valid = MultiTaskTensorDataset(X_valid, y_valid)
        self.test = MultiTaskTensorDataset(X_test, y_test)

    @staticmethod
    def __download__(root):
        logging.info("Downloading Census dataset")
        path = Path.joinpath(root, "census")
        census_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/"

        print(path.exists())
        if not path.exists():
            print(f"creating dirs for {str(path)}")
            path.mkdir(parents=True, exist_ok=True)

        if not os.path.exists("%s/census-income.data" % str(path)):
            print("Downloading train dataset")
            _download(
                url="%s/census-income.data.gz" % census_url,
                destination="%s/census-income.data.gz" % str(path),
                description="Train file download",
            )
            unzip("%s/census-income.data.gz" % str(path), "%s/census-income.data" % str(path))
            os.remove("%s/census-income.data.gz" % str(path))

        if not os.path.exists("%s/census-income.test" % str(path)):
            print("Downloading test dataset")
            _download(
                url="%s/census-income.test.gz" % census_url,
                destination="%s/census-income.test.gz" % str(path),
                description="Test file download",
            )
            unzip("%s/census-income.test.gz" % str(path), "%s/census-income.test" % str(path))
            os.remove("%s/census-income.test.gz" % str(path))


class LinCensusDataModule(CensusDataModule):
    """DataModule for Census dataset [1] as used in [2]. The tasks considered are income and never married.

    References:
        [1] R. Kohavi, “Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid,” in Proceedings of
            the second international conference on knowledge discovery and data mining (KDD-96), 1996.

        [2] X. Lin, H.-L. Zhen, Z. Li, Q. Zhang, and S. Kwong, “Pareto Multi-Task Learning,” in Advances in Neural
            Information Processing Systems, 2019.
    """

    def __init__(
        self,
        income: bool = True,
        age: bool = False,
        education: bool = True,
        never_married: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            income=income,
            age=age,
            education=education,
            never_married=never_married,
            *args,
            **kwargs,
        )

    @property
    def task_names(self) -> List[str]:
        return ["income", "education", "never_married"]

    @property
    def name(self) -> str:
        return "census-lin"

    @property
    def num_tasks(self):
        return 3

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.BINARY_CLASSIFICATION] * 3


class MaCensusDataModule(CensusDataModule):
    """DataModule for Census dataset [1] as used in [2]. The tasks considered are age, education and never married.

    References:
        [1] R. Kohavi, “Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid,” in Proceedings of
            the second international conference on knowledge discovery and data mining (KDD-96), 1996.

        [2] P. Ma, T. Du, and W. Matusik, “Efficient continuous pareto exploration in multi-task learning,”
            in International Conference on Machine Learning, 2020.
    """

    def __init__(
        self,
        income=False,
        age=True,
        education=True,
        never_married=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            income=income,
            age=age,
            education=education,
            never_married=never_married,
            *args,
            **kwargs,
        )

    @property
    def task_names(self) -> List[str]:
        return ["age", "education", "never_married"]

    @property
    def name(self) -> str:
        return "census-ma"

    @property
    def num_tasks(self):
        return 3

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.BINARY_CLASSIFICATION] * 3


class TestCensusDataModule(CensusDataModule):
    """Generic DataModule for Census dataset [1]. The datamodule currently supports the tasks income, age, education
    and never married.

    References:
        [1] R. Kohavi, “Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid,” in Proceedings of
            the second international conference on knowledge discovery and data mining (KDD-96), 1996.
    """

    def __init__(
        self,
        income: bool,
        age: bool,
        education: bool,
        never_married: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            income=income,
            age=age,
            education=education,
            never_married=never_married,
            *args,
            **kwargs,
        )

    @property
    def num_features(self):
        assert self.num_tasks == 2, "TestCensusDataModule only supports 2 tasks for the moment."
        if self.age and self.education:
            return 468
        if self.income and self.never_married:
            return 477
        if self.income and self.age:
            return 483

        raise NotImplementedError

    @property
    def task_names(self) -> List[str]:
        task_names = []
        if self.income:
            task_names.append("income")
        if self.age:
            task_names.append("age")
        if self.education:
            task_names.append("education")
        if self.never_married:
            task_names.append("never_married")
        return task_names

    @property
    def name(self) -> str:
        name = "census"
        if self.income:
            name += "_income"
        if self.age:
            name += "_age"
        if self.education:
            name += "_education"
        if self.never_married:
            name += "_never_married"
        return name

    @property
    def num_tasks(self):
        return int(self.income) + int(self.age) + int(self.education) + int(self.never_married)

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.BINARY_CLASSIFICATION] * self.num_tasks


###############################################################################
################################## UTILITIES ##################################
###############################################################################


def drop_columns(data):
    """
    Drops the unimportant columns from the original data
    :param data: a pandas data frame of the original data
    :return: a pandas data frame with only the important columns
    """
    output_data = data.drop(["MIGMTR1", "MIGMTR3", "MIGMTR4", "YEAR"], axis=1)

    return output_data


def drop_rows(data):
    """
    Drops rows having missing or NA values from the data set
    The training set is huge, so dropping few rows will not affect the learning
    ability of the classifiers
    :param data: a pandas data frame of the original data
    :return: a pandas data frame without any missing or NA value
    """
    row2 = np.where(data["HHDFMX"] == " Grandchild <18 ever marr not in subfamily")[0]
    data = data.drop(data.index[row2])

    return data


def drop_columns_and_rows(data):
    data = drop_columns(data)
    data = drop_rows(data)

    return data


def encode_binary_columns(data_train, data_test, list_of_binary_columns):
    label_encoder = LabelEncoder()
    for col in list_of_binary_columns:
        label_encoder.fit(data_train[col])
        data_train[col] = label_encoder.transform(data_train[col])
        data_test[col] = label_encoder.transform(data_test[col])

    return data_train, data_test


def standard_scaling_training_testing(train, test):
    scaler_features = preprocessing.StandardScaler()
    scaler_features.fit(train)

    train, test = scaler_features.transform(train), scaler_features.transform(test)

    return train, test, scaler_features


def task_education(x):
    if x.strip().startswith(("Bachelors", "Some", "Maters", "Asso", "Doctorate", "Prof")):
        return 1
    else:
        return 0


def task_income(x):
    if x == " - 50000.":
        return 0
    else:
        return 1


def task_nevermarried(x):
    if x == " Never married":
        return 1
    else:
        return 0


def task_age(x):
    if x >= 40:
        return 1
    else:
        return 0


def data_Census_preprocessing(
    root,
    income=False,
    age=False,
    education=False,
    never_married=False,
):
    # assert int(income) + int(age) + int(education) + int(never_married) >= 2, "Should have at least two tasks"
    root = Path(root)
    PATH_TO_TRAIN_DATASET = Path(root, "census", "census-income.data")
    PATH_TO_TEST_DATASET = Path(root, "census", "census-income.test")

    if not PATH_TO_TRAIN_DATASET.exists():
        CensusDataModule.__download__(root=root)

    print(root)
    df_testing = pd.read_csv(PATH_TO_TEST_DATASET, header=None, delimiter=",")
    df_training = pd.read_csv(PATH_TO_TRAIN_DATASET, header=None, delimiter=",")

    df_training.columns = [
        "AAGE",
        "ACLSWKR",
        "ADTIND",
        "ADTOCC",
        "AHGA",
        "AHRSPAY",
        "AHSCOL",
        "AMARITL",
        "AMJIND",
        "AMJOCC",
        "ARACE",
        "AREORGN",
        "ASEX",
        "AUNMEM",
        "AUNTYPE",
        "AWKSTAT",
        "CAPGAIN",
        "CAPLOSS",
        "DIVVAL",
        "FILESTAT",
        "GRINREG",
        "GRINST",
        "HHDFMX",
        "HHDREL",
        "MARSUPWT",
        "MIGMTR1",
        "MIGMTR3",
        "MIGMTR4",
        "MIGSAME",
        "MIGSUN",
        "NOEMP",
        "PARENT",
        "PEFNTVTY",
        "PEMNTVTY",
        "PENATVTY",
        "PRCITSHP",
        "SEOTR",
        "VETQVA",
        "VETYN",
        "WKSWORK",
        "YEAR",
        "target",
    ]
    df_testing.columns = df_training.columns

    task_names = []
    labels = []
    if income:
        task_names.append("target")
        labels.append("income_encoded")
        # Transform target to binary problem
        df_training["income_encoded"] = df_training["target"].apply(task_income)
        df_training = df_training.drop(columns=["target"])

        df_testing["income_encoded"] = df_testing["target"].apply(task_income)
        df_testing = df_testing.drop(columns=["target"])

    if age:
        task_names.append("AAGE")
        labels.append("age_encoded")
        # Transform Marital status to binary problem
        df_training["age_encoded"] = df_training["AAGE"].apply(task_age)
        df_training = df_training.drop(columns=["AAGE"])

        df_testing["age_encoded"] = df_testing["AAGE"].apply(task_age)
        df_testing = df_testing.drop(columns=["AAGE"])

    if education:
        task_names.append("AHGA")
        labels.append("education_encoded")
        # Transform Marital status to binary problem
        df_training["education_encoded"] = df_training["AHGA"].apply(task_education)
        df_training = df_training.drop(columns=["AHGA"])

        df_testing["education_encoded"] = df_testing["AHGA"].apply(task_education)
        df_testing = df_testing.drop(columns=["AHGA"])

    if never_married:
        task_names.append("AMARITL")
        labels.append("married_encoded")
        # Transform Marital status to binary problem
        df_training["married_encoded"] = df_training["AMARITL"].apply(task_nevermarried)
        df_training = df_training.drop(columns=["AMARITL"])

        df_testing["married_encoded"] = df_testing["AMARITL"].apply(task_nevermarried)
        df_testing = df_testing.drop(columns=["AMARITL"])

    # Clean dataset
    df_training = drop_columns_and_rows(df_training)
    df_testing = drop_columns_and_rows(df_testing)

    continuous_columns = list(
        set(["AAGE", "AHRSPAY", "DIVVAL", "CAPGAIN", "CAPLOSS", "WKSWORK", "MARSUPWT"]) - set(task_names)
    )
    binary_columns = ["ASEX"]

    print(f"The labels are {labels}")

    # List of names of columns with dummy features
    dummy_columns = [
        col
        for col in df_training.columns.values
        if col not in continuous_columns
        if col not in binary_columns
        if col not in labels
    ]

    df_training, df_testing = encode_binary_columns(df_training, df_testing, binary_columns)

    df_training_dummy = pd.get_dummies(df_training[dummy_columns], columns=dummy_columns)
    df_testing_dummy = pd.get_dummies(df_testing[dummy_columns], columns=dummy_columns)

    X_continous_train, X_continous_test, _ = standard_scaling_training_testing(
        df_training[continuous_columns].values.astype("float64"),
        df_testing[continuous_columns].values.astype("float64"),
    )
    X_train = np.concatenate(
        (X_continous_train, df_training_dummy.values, df_training[binary_columns].values), axis=1
    ).astype("float64")
    X_test = np.concatenate(
        (X_continous_test, df_testing_dummy.values, df_testing[binary_columns].values), axis=1
    ).astype("float64")

    y_train = df_training[labels].values.astype("long")
    y_test = df_testing[labels].values.astype("long")

    X_train = torch.from_numpy(X_train).type("torch.FloatTensor")
    X_test = torch.from_numpy(X_test).type("torch.FloatTensor")
    y_train = torch.from_numpy(y_train).type("torch.LongTensor")
    y_test = torch.from_numpy(y_test).type("torch.LongTensor")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    CensusDataModule.__download__(root="~/benchmarks/data/")
