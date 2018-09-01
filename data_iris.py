from sklearn.datasets import load_iris
import pandas as pd

__setosa_category = 0
__versicolor_category = 1
__virginica_category = 2

__iris_data_set = load_iris(True)

__all_data = pd.DataFrame(__iris_data_set[0])
__all_targets = pd.DataFrame(__iris_data_set[1])


def __categories_transform_func(c):
    return 1 if c == 0 else -1


def get_iris_training_set():
    """
    Gets iris data set for training purposes
    :return: tuple with format (iris_features, target_variables)
    """
    setosa_training_count = 25
    not_setosa_training_count = 75

    all_training_data = pd.concat([__all_data.iloc[:setosa_training_count], __all_data[-not_setosa_training_count:]])
    all_training_target_data = pd.concat([__all_targets.iloc[:setosa_training_count],
                                          __all_targets.iloc[-not_setosa_training_count:]])

    transformed_targets = all_training_target_data.applymap(__categories_transform_func).T.iloc[0]

    return all_training_data, transformed_targets


def get_iris_test_set():
    """
    Get iris data set for testing purposes
    :return: tuple with format (iris_features, target_variables)
    """
    from_idx = 25
    to_idx = 75

    all_test_data = __all_data.iloc[from_idx:to_idx]
    all_test_target_data = __all_targets.iloc[from_idx:to_idx]
    transformed_targets = all_test_target_data.applymap(__categories_transform_func)

    return all_test_data, transformed_targets