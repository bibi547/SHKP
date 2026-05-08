import torch
import json
import torch.nn as nn
import os
from glob import glob
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as T

import os
os.environ["XFORMERS_FORCE_DISABLE"] = "1"

ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel",}
NAMES2ID = {v: k for k, v in ID2NAMES.items()}


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_name = 'dinov2_vitg14'
    model = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=False).cuda()
    weights_path = './dinov2_vitg14_pretrain.pth'
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    root_dir = "./datasets/KeypointNet"
    class_name = 'airplane'
    img_root = os.path.join(root_dir, 'images', NAMES2ID[class_name])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    for model_id in os.listdir(img_root):
        model_dir = os.path.join(img_root, model_id)
        for i in range(62):
            img_file = os.path.join(model_dir, f"view_{i}.png")
            mask_file = os.path.join(model_dir, f"view_{i}_mask.png")
            img = Image.open(img_file).convert('RGB')
            image_tensor = transform(img).unsqueeze(0).cuda()
            mask = Image.open(mask_file).convert("L")
            mask = T.ToTensor()(mask)  # (1,H,W), 0 or 1

            with torch.no_grad():
                out = model.forward_features(image_tensor)
                x_patch = out['x_norm_patchtokens']  # (1, 256, 1536)
            feature_map = x_patch.squeeze().cpu()

            save_file = os.path.join(model_dir, f"view_{i}_dino.pt")
            torch.save(feature_map, save_file)
            print(model_id + 'view' + str(i))












