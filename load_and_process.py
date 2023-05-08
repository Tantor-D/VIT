import urllib

import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm


# if __name__ == "__main__":
#     config = resolve_data_config({}, model="vit_base_patch16_224")
#     transform = create_transform(**config)
#
#     out =


if __name__ == "__main__":


    model = timm.create_model('vit_base_patch16_224',  num_classes=0 ,pretrained=True)
    model.eval()
    out = model(torch.randn(2, 3, 224, 224))
    print(out.shape)
