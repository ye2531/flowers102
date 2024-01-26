
from typing import Tuple, Dict, List
import pandas as pd

def get_categories(filepath: str) -> Tuple[List[str], Dict[str, int]]:
    """Generates list of flower names and necessary mappings from CSV file.

    Takes in CSV file containing (flower name, numeric label) pairs. Generates a list of flower names,
    a dictionary that maps flower name to numeric label and a dictionary that maps numeric label of flower to
    corresponding id.

    Args:
        filepath: A path to CSV file containing (flower name, numeric label) pairs.
    """

    df = pd.read_csv(filepath)
    categories = df["category"].tolist()
    label_to_category = {record["label"]: record["category"] for record in df.to_dict("records")}
    label_id_to_label = {i: label for i, label in enumerate(label_to_category.keys())}

    return categories, label_to_category, label_id_to_label

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def create_writer(experiment_name: str) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance.

    An experiment name is a string of format runs/timestamp/experiment_name/.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter()
    """

    timestamp = datetime.now().strftime("%Y-%m-%d")

    experiment_dir = os.path.join("runs", timestamp, experiment_name)

    return SummaryWriter(log_dir=experiment_dir)
