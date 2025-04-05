import os
import torch
from torch import nn
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from pathlib import Path

class EfficientVit(nn.Module):
    def __init__(self, weitghts_name="efficientvit_l3_r320"):
        super().__init__()
        base_path = Path(os.environ["PYTHONPATH"])
        weights_root = "models/efficient_vit/"
        if not weitghts_name.endswith(".pt"):
            weitghts_name += ".pt"
        weight_url = base_path.joinpath(weights_root, weitghts_name)
        model_name = Path(weitghts_name).stem.replace("_", "-")
        self.model = create_efficientvit_cls_model(
            name=model_name,
            weight_url=weight_url
        )

if __name__ == "__main__":
    model = EfficientVit()
    model.model.head = nn.Identity()
    print(model)