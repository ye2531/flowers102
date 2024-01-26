
import requests
import tarfile
import os
import scipy
import pandas as pd
from typing import Tuple


ARCHIVE_URL =  "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
IMAGE_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
SET_ID_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"


def download_and_extract_data(archive_url: str,
                              image_labels_url: str,
                              set_id_url: str,
                              remove_source: bool = True) -> Tuple[str, ...]:
    """Downloads dataset of images, image labels and dataset split indices from source urls.

    Args:
        archive_url (str): A link to a TAR file containing image data.
        image_labels_url (str): A link to a MAT file containing image to label mapping.
        set_id_url (str): A link to a MAT file containing dataset split to image indices mapping.
        remove_source (bool):  Whether to remove the source files after downloading and extracting the data.

    Returns:
        Tuple[str, ...]: Paths to downloaded data folder and files.
    """

    # Download TAR archive file containing the data
    response = requests.get(archive_url)
    archive_path = os.path.join(os.getcwd(), "oxfordflower102.tar")

    with open(archive_path, 'wb') as tar_file:
        tar_file.write(response.content)

    with tarfile.open(archive_path, 'r:gz') as tar:
        image_folder_name = [member for member in tar.getmembers() if member.isdir()][0].name
        tar.extractall()

    image_path = os.path.join(os.getcwd(), image_folder_name)

    if remove_source:
        os.remove(archive_path)

    # Download MAT file containing image labels
    response = requests.get(image_labels_url)
    labels_path = os.path.join(os.getcwd(), "image_labels.mat")

    with open(labels_path, 'wb') as mat_file:
        mat_file.write(response.content)

    # Download MAT file containing data split ids
    response = requests.get(set_id_url)
    set_id_path = os.path.join(os.getcwd(), "set_id.mat")

    with open(set_id_path, 'wb') as mat_file:
        mat_file.write(response.content)

    return image_path, labels_path, set_id_path


def split_data(image_path: str,
               set_id_path: str,
               data_path: str):
    """Splits image data into train, valid and test directories.

    Takes in a path to MAT file containing dataset split to image indices mapping. Moves images
    from source directory to train, valid and test directories in accordance with said mapping.

    Args:
        image_path (str): A path to the source directory containing image data.
        set_id_path (str): A path to MAT file containing dataset split to image indices mapping.
        data_path (str): A destination data path.

    Returns:
        dict: A dictionary of dataset split indices.
    """

    new_image_path = os.path.join(data_path, "images")
    os.mkdir(new_image_path)

    splits_mat = scipy.io.loadmat(set_id_path)
    splits = {
        "train": splits_mat["tstid"][0],
        "valid": splits_mat["valid"][0],
        "test": splits_mat["trnid"][0]
    }

    for split, ids in splits.items():
        split_path = os.path.join(new_image_path, split)
        os.mkdir(split_path)

        for id in ids:
            old_file_path = os.path.join(image_path, f"image_{id:05d}.jpg")
            new_file_path = os.path.join(split_path, f"image_{id:05d}.jpg")
            os.rename(old_file_path, new_file_path)

    return splits


def extract_labels(labels_path: str, data_path: str, splits: dict):
    """Extracts labels from MAT file, generates CSV file containing image filename-label-split triples.

    Args:
        labels_path (str): A path to MAT file containing image to label mapping.
        data_path (str): A destination data path.
        splits (dict): A dictionary where keys are split names and values are image indices belonging to that split.
    """

    labels_mat = scipy.io.loadmat(labels_path)
    num_images = labels_mat['labels'].shape[1]
    labels = labels_mat['labels'][0].tolist()

    image_file_pattern = "image_{:05d}.jpg"
    image_strings = [image_file_pattern.format(i) for i in range(1, num_images + 1)]

    df = pd.DataFrame({"image_file_name": image_strings,
                       "label": labels})

    df['split'] = ''

    for split, ids in splits.items():
        df.loc[ids-1, "split"] = split

    df.to_csv(os.path.join(data_path, "image_to_label.csv"), index=False)


def setup_data(archive_url: str = ARCHIVE_URL,
               image_labels_url: str = IMAGE_LABELS_URL,
               set_id_url: str = SET_ID_URL,
               remove_source: bool = True):
    """Downloads and exctracts the data from URLs, then splits it into train, valid and test sets.

    Takes in URLs, downloads archives images and MAT files containing image to label mapping
    and dataset split to image indices mapping. Splits dataset of images into train, valid and test
    directories in accordance with said mapping.

    Args:
        archive_url (str): A link to a TAR file containing image data.
        image_labels_url (str): A link to a MAT file containing image to label mapping.
        set_id_url (str): A link to a MAT file containing dataset split to image indices mapping.
        remove_source (bool):  Whether to remove the source files after downloading and extracting the data.

    Returns:
        str: A path to downloaded and splitted data.
    """

    data_path = os.path.join(os.getcwd(), "data")

    if os.path.isdir(data_path):
        print(print(f"[INFO] {data_path}/ directory exists, skipping download."))
    else:
        print(f"[INFO] Did not find {data_path}/ directory, downloading the data...")
        os.mkdir(data_path)

        image_path, labels_path, set_id_path = download_and_extract_data(archive_url=archive_url,
                                                                         image_labels_url=image_labels_url,
                                                                         set_id_url=set_id_url,
                                                                         remove_source=remove_source)
        splits = split_data(image_path=image_path,
                            set_id_path=set_id_path,
                            data_path=data_path)

        extract_labels(labels_path=labels_path, data_path=data_path, splits=splits)

        if remove_source:
            os.remove(labels_path)
            os.remove(set_id_path)
            os.rmdir(image_path)

        print(f"[INFO] Data path: {data_path}")
        return data_path
