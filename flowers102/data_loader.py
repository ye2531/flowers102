
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader


NUM_WORKERS = os.cpu_count()


def create_dataset(train_dir: str,
                   test_dir: str,
                   image_to_label_path: str,
                   category_path: str = None,
                   train_transform: torchvision.transforms = None,
                   test_transform: torchvision.transforms = None):
    """Creates PyTorch Datasets from raw image data.

    Args:
        train_dir (str): A path to training data.
        test_dir (str): A path to test data.
        image_to_label_path (str): A path to CSV file containing image filename-label-split triples.
        category_path (str): A path to CVS file containing flower name to numeric label mapping (optional).
        train_transform: torchvision transforms to perform on training data (optional).
        test_transform: torchvision transforms to perform on testing data (optional).

    Returns:
        Tuple[torch.utils.data.Dataset, ...]
    """

    train_data = ImageFolderCustom(target_dir=train_dir,
                                   image_to_label_path=image_to_label_path,
                                   category_path=category_path,
                                   transform=train_transform)

    test_data = ImageFolderCustom(target_dir=test_dir,
                                  image_to_label_path=image_to_label_path,
                                  category_path=category_path,
                                  transform=test_transform)


    return train_data, test_data


def get_n_pct_subset(train_data: torch.utils.data.Dataset,
                     subset_size: float,
                     test_data: torch.utils.data.Dataset = None,
                     seed: int = 31):
    """Generates a subset of PyTorch Dataset of specified size.

    Args:
        train_data: PyTorch Dataset containing training data.
        subset_size (float): Size of subset to generate, value from 0 to 1.
        test_data: PyTorch Dataset containing testing data (optional).
        seed (int): Random seed to set. Defaults to 31.

    Returns:
        List[torch.utils.data.Dataset, ...]
    """

    subsets = []

    train_targets = train_data.targets

    train_indices, _, _, _, = train_test_split(range(len(train_data)),
                                               train_targets,
                                               stratify=train_targets,
                                               train_size=subset_size,
                                               random_state=seed)

    train_data_subset = Subset(dataset=train_data, indices=train_indices)
    subsets.append(train_data_subset)

    if test_data:
        test_targets = test_data.targets

        test_indices, _, _, _, = train_test_split(range(len(test_data)),
                                                  test_targets,
                                                  stratify=test_targets,
                                                  train_size=subset_size,
                                                  random_state=seed)

        test_data_subset = Subset(dataset=test_data, indices=test_indices)
        subsets.append(test_data_subset)
        return subsets

    return subsets


def create_dataloaders(train_data: torch.utils.data.Dataset,
                       test_data: torch.utils.data.Dataset,
                       batch_size: int,
                       num_workers: int = NUM_WORKERS):
    """Creates PyTorch DataLoaders from PyTorch Datasets.

    Takes in train and test datasets and turns them into PyTorch DataLoaders with specified batch size.

    Args:
        train_data: PyTorch Dataset containing training data.
        test_data: PyTorch Dataset containing testing data (optional).
        batch_size: Number of samples per batch in the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        Tuple[torch.utils.data.DataLoader, ...]
    """

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)

    return train_dataloader, test_dataloader


def load_data(train_dir: str,
              test_dir: str,
              image_to_label_path: str,
              batch_size: int,
              num_workers: int = NUM_WORKERS,
              category_path: str = None,
              train_transform: torchvision.transforms = None,
              test_transform: torchvision.transforms = None):
    """Creates PyTorch DataLoaders from raw image data.

    Args:
        train_dir (str): A path to training data.
        test_dir (str): A path to test data.
        image_to_label_path (str): A path to CSV file containing image filename-label-split triples.
        batch_size: Number of samples per batch in the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        valid_dir (str): A path to validation data (optional).
        category_path (str): A path to CVS file containing flower name to numeric label mapping (optional).
        train_transform: torchvision transforms to perform on training data (optional).
        test_transform: torchvision transforms to perform on testing data (optional).
        valid_transform: torchvision transforms to perform on validation data (optional).

    Returns:
        Tulpe[torch.utils.data.DataLoader, ...]
    """


    train_data, test_data = create_dataset(train_dir=train_dir,
                                           test_dir=test_dir,
                                           image_to_label_path=image_to_label_path,
                                           category_path=category_path,
                                           train_transform=train_transform,
                                           test_transform=test_transform)

    train_dataloader, test_dataloader = create_dataloaders(train_data=train_data,
                                                           test_data=test_data,
                                                           batch_size=batch_size,
                                                           num_workers=num_workers)
    
    return train_dataloader, test_dataloader
