from enum import Enum


class TaskCategoriesEnum(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    DEPTH_REGRESSION = "depth_regression"
