import os
import torch
from torch import nn
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from pathlib import Path
from transformers import ViTImageProcessor
import re

class EfficientVit(nn.Module):
    def __init__(self, weitghts_name="efficientvit_l3_r320", device="cuda"):
        super().__init__()
        self.device = device
        self.build_processor(weitghts_name)
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
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.model.head = nn.Identity()
        self.model = self.model.to(device=device)
    
    def build_processor(self, model_name):
        size = re.findall(r"_r(\d+)", model_name)[0]
        self._processor = ViTImageProcessor(
            size={"width": size, "height": size}
        )
    
    def forward(self, image):
        image = image.to(self.device)
        emb = self.model(image)["stage_final"]
        emb = self.gap(emb)
        return emb
    
    @property
    def processor(self):
        return self._processor

if __name__ == "__main__":
    model = EfficientVit()
    model.eval()
    x = torch.rand(1, 3, 320, 320)
    out = model(x)
    print(out.shape)