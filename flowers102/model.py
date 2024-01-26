
import torchvision
from torch import nn

def create_model(num_classes: int):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): Number of classes in the classifier head. Defaults to 3.
        seed (int, optional): Random seed value. Defaults to 31.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model.
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transform = weights.transforms()

    try:
        model = torchvision.models.efficientnet_b2(weights=weights)
    except RuntimeError as e:
        if "invalid hash value" in str(e):

            from torchvision.models._api import WeightsEnum
            from torch.hub import load_state_dict_from_url

            def get_state_dict(self, *args, **kwargs):
                kwargs.pop("check_hash")
                return load_state_dict_from_url(self.url, *args, **kwargs)

            WeightsEnum.get_state_dict = get_state_dict

            model = torchvision.models.efficientnet_b2(weights=weights)
        else:
            raise

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return model, transform
