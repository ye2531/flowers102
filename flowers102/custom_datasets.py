
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from typing import Tuple
import os

class ImageFolderCustom(Dataset):
    """Creates custom dataset from raw image data.

    Args:
        target_dir (str): A path to image data directory.
        image_to_label_path (str): A path to CSV file containing image filename-label-split triples.
        category_path (str): A path to CVS file containing flower name to numeric label mapping (optional).
        transform:  torchvision transforms to perform on training and testing data (optional).
    """
    def __init__(self,
                 target_dir: str,
                 image_to_label_path: str,
                 category_path: str = None,
                 transform: torchvision.transforms = None) -> None:

        self.image_to_label_df = pd.read_csv(image_to_label_path)
        self.target_dir = target_dir
        self.split = os.path.basename(target_dir)
        self.targets = self.image_to_label_df[self.image_to_label_df['split'] == self.split].iloc[:, 1].tolist()
        self.transform = transform

        if category_path:
            self.categories, self.label_to_category, self.label_id_to_label = get_categories(category_path)

    # Overwrite the __len__() method
    def __len__(self) -> int:
        return len(os.listdir(self.target_dir))

    # Overwrite the __getitem__() method
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_filename = self.image_to_label_df[self.image_to_label_df['split'] == self.split].iloc[index, 0]
        label = self.image_to_label_df[self.image_to_label_df['split'] == self.split].iloc[index, 1]

        try:
            category = self.label_to_category[label]
            label_id = self.categories.index(category)
        except AttributeError:
            label_id = label - 1

        image = Image.open(os.path.join(self.target_dir, image_filename))

        # Transform if necessary
        if self.transform:
            return self.transform(image), label_id
        else:
            return image, label_id
