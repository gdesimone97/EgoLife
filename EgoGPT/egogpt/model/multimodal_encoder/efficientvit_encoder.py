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
        self.weitghts_name = weitghts_name
        self.build_processor(weitghts_name)
        self.model = self.load_model()
    
    def build_processor(self, model_name):
        size = re.findall(r"_r(\d+)", model_name)[0]
        size={"width": 336, "height": 336}
        self._processor = ViTImageProcessor(
            size=size
        )
    
    def forward(self, image):
        image = image.to(self.device)
        emb = self.model(image)["stage_final"]
        emb = self.gap(emb)
        emb = emb.view(emb.size(0), -1, emb.size(-3))
        return emb
    
    def load_model(self):
        base_path = Path(os.environ["PYTHONPATH"])
        weights_root = "models/efficient_vit/"
        if not self.weitghts_name.endswith(".pt"):
            self.weitghts_name += ".pt"
        weight_url = base_path.joinpath(weights_root, self.weitghts_name)
        model_name = Path(self.weitghts_name).stem.replace("_", "-")
        self.model = create_efficientvit_cls_model(
            name=model_name,
            weight_url=weight_url
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.model.head = nn.Identity()
        return self.model
    
    @property
    def image_processor(self):
        return self._processor
    
    @property
    def config(self):
        return {
            "mm_projector_type": "mlp2x_gelu",
            "mm_hidden_size": 1024
        }

    @property
    def hidden_size(self):
        return self.config["mm_hidden_size"]
    
    @property
    def num_patches_per_side(self):
        return 27
    
if __name__ == "__main__":
    model = EfficientVit()
    model.eval()
    x = torch.rand(1, 3, 320, 320)
    out = model(x)
    print(out.shape)