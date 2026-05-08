import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets.utils import geodesic_heatmaps
from datasets.sh_util import compute_vis_stats_from_heatmaps
import matplotlib.pyplot as plt

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

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class KeypointNet_Dataset(Dataset):
    def __init__(self, args, train: bool):
        self.args = args
        self.class_id = NAMES2ID[args.class_name]
        self.transform = make_transform(resize_size=224)
        if train:
            split_file = args.train_file
        else:
            split_file = args.test_file

        annots = json.load(open(args.anno_dir))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[args.class_name]]
        keypoints = dict([(annot['model_id'],[(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1

        split_models = []
        with open(os.path.join(args.split_root, split_file)) as f:
            for line in f:
                cls_id, model_id = line.strip().split('-', 1)
                if cls_id == self.class_id:
                    split_models.append(model_id)

        self.samples = []
        for model_id in split_models:
            mv_path = os.path.join(args.img_root, self.class_id, model_id)
            kps = keypoints[model_id]
            kp_classes = -np.ones((self.nclasses,), dtype=np.int64)
            for i, kp in enumerate(kps):
                kp_classes[i] = kp[1]
            kp_classes += 1

            for i in range(62):
                img_file = os.path.join(mv_path, f"view_{i}.png")
                mask_file = os.path.join(mv_path, f"view_{i}_mask.png")
                dino_file = os.path.join(mv_path, f"view_{i}_dino.pt")
                dist_file = os.path.join(mv_path, f"view_{i}_dist.pt")
                if os.path.exists(img_file) and os.path.exists(mask_file) and os.path.exists(dist_file):
                    self.samples.append((kp_classes, img_file, mask_file, dino_file, dist_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kp_classes, img_file, mask_file, dino_file, dist_file = self.samples[idx]

        # ---- load image (RGB) ----
        img = Image.open(img_file).convert("RGB")
        img = T.ToTensor()(img)  # (3,H,W), normalized to [0,1]
        # img = self.transform(img)

        # ---- load dino feature ----
        dino_emb = torch.load(dino_file, weights_only=True)  # (patch_num=256, C)
        H = W = int(dino_emb.shape[0] ** 0.5)
        C = dino_emb.shape[1]
        dino_feat = dino_emb.reshape(H, W, C)  # (16, 16, 1536)
        dino_feat = dino_feat.permute(2, 0, 1)  # (1536, 16, 16)

        # ---- load mask ----
        mask = Image.open(mask_file).convert("L")
        mask = T.ToTensor()(mask)  # (1,H,W), 0 or 1
        mask_bool = (mask > 0.5).squeeze(0)  # (H,W) boolean

        # ---- load distance map ----
        dist = torch.load(dist_file, weights_only=True)  # (H, W, K)
        dist = dist.permute(2, 0, 1).float()  # (K, H, W)
        heatmaps = geodesic_heatmaps(dist, self.args)
        heatmaps = torch.from_numpy(heatmaps).float()
        heatmaps = heatmaps * mask.expand_as(heatmaps)
        vis_states = compute_vis_stats_from_heatmaps(heatmaps, mask, area_thresh=0.9)

        kp_num = heatmaps.shape[0]
        heats = -torch.ones(self.nclasses, heatmaps.shape[1], heatmaps.shape[2]).float()
        heats[:kp_num, :, :] = heatmaps
        vis_scores = torch.zeros(self.nclasses).float()
        vis_scores[:kp_num] = vis_states

        kp_classes = torch.from_numpy(kp_classes).long()

        return kp_classes, img, dino_emb, dino_feat, mask, heats, vis_scores

class KPSH_Dataset(Dataset):
    def __init__(self, args, train: bool):
        self.args = args
        self.class_id = NAMES2ID[args.class_name]
        self.transform = make_transform(resize_size=224)
        if train:
            split_file = args.train_file
        else:
            split_file = args.val_file

        annots = json.load(open(args.anno_dir))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[args.class_name]]
        keypoints = dict([(annot['model_id'],[(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1

        split_models = []
        with open(os.path.join(args.split_root, split_file)) as f:
            for line in f:
                cls_id, model_id = line.strip().split('-', 1)
                if cls_id == self.class_id:
                    split_models.append(model_id)

        self.samples = []
        for model_id in split_models:
            mv_path = os.path.join(args.img_root, self.class_id, model_id)
            kps = keypoints[model_id]
            kp_classes = -np.ones((self.nclasses,), dtype=np.int64)
            for i, kp in enumerate(kps):
                kp_classes[i] = kp[1]
            kp_classes += 1

            for i in range(62):
                img_file = os.path.join(mv_path, f"view_{i}.png")
                mask_file = os.path.join(mv_path, f"view_{i}_mask.png")
                dino_file = os.path.join(mv_path, f"view_{i}_dino.pt")
                dist_file = os.path.join(mv_path, f"view_{i}_dist.pt")
                sh_file = os.path.join(mv_path, f"view_{i}_sh.npy")
                if os.path.exists(img_file) and os.path.exists(mask_file) and os.path.exists(dist_file):
                    self.samples.append((kp_classes, img_file, mask_file, dino_file, dist_file, sh_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kp_classes, img_file, mask_file, dino_file, dist_file, sh_file = self.samples[idx]

        # ---- load image (RGB) ----
        img = Image.open(img_file).convert("RGB")
        img = T.ToTensor()(img)  # (3,H,W), normalized to [0,1]

        # ---- load dino feature ----
        dino_emb = torch.load(dino_file, weights_only=True).float()     # (patch_num=256, C)
        H = W = int(dino_emb.shape[0] ** 0.5)
        C = dino_emb.shape[1]
        dino_feat = dino_emb.reshape(H, W, C)  # (16, 16, 1536)
        dino_feat = dino_feat.permute(2, 0, 1)  # (1536, 16, 16)

        # ---- load mask ----
        mask = Image.open(mask_file).convert("L")
        mask = T.ToTensor()(mask)  # (1,H,W), 0 or 1
        mask_bool = (mask > 0.5).squeeze(0)  # (H,W) boolean

        # ---- load distance map ----
        dist = torch.load(dist_file, weights_only=True)  # (H, W, K)
        dist = dist.permute(2, 0, 1).float()  # (K, H, W)
        heatmaps = geodesic_heatmaps(dist, self.args)
        heatmaps = torch.from_numpy(heatmaps).float()
        heatmaps = heatmaps * mask.expand_as(heatmaps)

        kp_num = heatmaps.shape[0]
        heats = -torch.ones(self.nclasses, heatmaps.shape[1], heatmaps.shape[2]).float()
        heats[:kp_num, :, :] = heatmaps

        sh_scores = np.load(sh_file)
        sh_scores = torch.from_numpy(sh_scores).float()

        kp_classes = torch.from_numpy(kp_classes).long()

        return kp_classes, img, dino_emb, dino_feat, mask, heats, sh_scores

