import os
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
import tqdm
import trimesh
import numpy as np
from PIL import Image
import torchvision.transforms as T

from pl_model import LitModel
from datasets.keypointnet_data import KeypointNet_Dataset, NAMES2ID
from utils.load_utils import load_mesh, remove_unreferenced_vertices
from utils.rendering import setup_renderer, sample_view_points
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import matplotlib.pyplot as plt
from datasets.utils import geodesic_heatmaps
from utils.vis_utils import (raster_fragments_for_view,cluster_predicted_keypoints)
from datasets.utils import naive_read_pcd
from utils.eval_utils import get_cd, get_cd_dist, hungary_iou, get_iou, get_geo_iou
from scipy.spatial.distance import cdist


def inference_and_map_to_mesh(checkpoint,
                              test_file='./datasets/KeypointNet/splits/test.txt',
                              root_dir='./datasets/KeypointNet',):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    model = LitModel.load_from_checkpoint(checkpoint).to(device)
    model.eval()
    args = model.hparams.args

    # load gt kps
    class_id = NAMES2ID[args.class_name]
    anno_file = './datasets/KeypointNet/annotations/all.json'
    annots = json.load(open(anno_file))
    annots = [annot for annot in annots if annot['class_id'] == class_id]
    keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])

    # dataset
    dataset = KeypointNet_Dataset(args, train=False)

    model_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, img_file, _, _, _ = dataset.samples[idx]
        model_dir = os.path.dirname(img_file)
        mesh_name = os.path.basename(model_dir)
        model_to_indices[mesh_name].append(idx)

    test_models = []
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id, model_id = line.split('-', 1)
            test_models.append(model_id)

    # setup renderer and view sampling (must match preprocessing)
    render_dist = 1.1
    renderer = setup_renderer(device)
    views = sample_view_points(render_dist, 5)  # (N_VIEWS, 3)
    view_dirs = views / np.linalg.norm(views, axis=1, keepdims=True)
    view_dirs_t = torch.from_numpy(view_dirs.astype(np.float32)).to(device)

    mcd = []
    miou = {}
    for i in range(11):
        key = i * 0.01
        if i == 0: key = 0.001
        miou[key] = []

    for mesh_name, indices in tqdm.tqdm(model_to_indices.items(), desc="models"):

        pcd_file = os.path.join(root_dir, 'pcds',class_id, mesh_name + '.pcd')
        geo_file = os.path.join(root_dir, 'pcds',class_id, mesh_name + '.txt')
        geo_dists = np.loadtxt(geo_file, delimiter=',')
        points = naive_read_pcd(pcd_file)[0]
        kps = keypoints[mesh_name]
        kps_xyz = []
        for i, kp in enumerate(kps):
            kp_xyz = points[kp[0]]
            kps_xyz.append(kp_xyz)
        gt_kps = np.array(kps_xyz)

        # load mesh (use the same loading as in preprocess)
        mesh_file = os.path.join(root_dir, 'ShapeNetCore.v2.ply', class_id, mesh_name + '.ply')
        mesh_p3d = load_mesh(mesh_file).to(device)
        faces = mesh_p3d.faces_packed()
        verts = mesh_p3d.verts_packed()
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
        keep = areas > 1e-8
        mesh_p3d = Meshes(verts=[verts], faces=[faces[keep]])
        mesh_p3d = remove_unreferenced_vertices(mesh_p3d)
        mesh_p3d.textures = TexturesVertex(verts_features=torch.ones_like(mesh_p3d.verts_packed()[None]) * 0.7)

        verts_np = mesh_p3d.verts_packed().detach().cpu().numpy()
        faces_np = mesh_p3d.faces_packed().detach().cpu().numpy()
        mesh_tm = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

        view_indices = sorted(model_to_indices[mesh_name])  # these correspond to dataset indices (per view)

        sem_weights = defaultdict(list)
        sem_xyz = defaultdict(list)

        for idx in view_indices:
            kp_classes, img_file, mask_file, dino_file, dist_file = dataset.samples[idx]

            # infer view id from filename "view_{i}.png"
            basename = os.path.basename(img_file)
            view_id = int(basename.split('_')[1].split('.')[0])
            # folder path
            folder = os.path.dirname(img_file)
            coord_file = os.path.join(folder, f"view_{view_id}_coord.pt")
            # bary_coords = torch.load(coord_file, map_location=device)

            # load image and gt heatmaps
            kp_classes = np.array(kp_classes)
            kp_classes = kp_classes[kp_classes != 0]
            img = Image.open(img_file).convert("RGB")
            img = T.ToTensor()(img)  # (3,H,W), normalized to [0,1]
            mask = Image.open(mask_file).convert("L")
            mask = T.ToTensor()(mask)  # (1,H,W), 0 or 1
            dist = torch.load(dist_file, weights_only=True)  # (H, W, K)
            dist = dist.permute(2, 0, 1).float()  # (K, H, W)
            gt_heats = geodesic_heatmaps(dist, args)
            gt_heats = torch.from_numpy(gt_heats).float()
            gt_heats = gt_heats * mask.expand_as(gt_heats)
            gt_heats = gt_heats.to(device)
            pix_to_face, bary_coords, fragments = raster_fragments_for_view(mesh_p3d, renderer, views, render_dist, view_id, device)

            # predict heatmaps
            dino_emb = torch.load(dino_file, weights_only=True)  # (256, C)
            H = int(dino_emb.shape[0] ** 0.5)
            C = dino_emb.shape[1]
            dino_emb_t = dino_emb.clone().detach().float().unsqueeze(0).to(device)
            dino_feat = dino_emb.reshape(H, H, C).permute(2, 0, 1)
            dino_feat_t = dino_feat.clone().detach().float().unsqueeze(0).to(device)
            img_t = img.clone().detach().float().unsqueeze(0).to(device)
            with torch.no_grad():
                probs, pred_heats, pred_vis = model(dino_emb_t,dino_feat_t)
                pred_heats = pred_heats * mask.cuda()
                pred_vis = pred_vis.squeeze()
            probs = probs[0]        # (M, num_classes)
            pred_heats = pred_heats[0]  # (M, H, W)
            pred_cls_ids = probs.argmax(dim=-1)  # (M,)
            fg_mask = pred_cls_ids > 0
            fg_idx = torch.nonzero(fg_mask, as_tuple=True)[0]
            if fg_idx.numel() == 0:
                continue
            pred_heats_fg = pred_heats[fg_idx]  # (K_pred, H, W)
            pred_vis_fg = pred_vis[fg_idx]

            # gt_heats: (K, H, W)
            K, H, W = pred_heats_fg.shape
            ph_flat = pred_heats_fg.view(K, -1)  # (K, H*W)

            max_vals, max_ids = torch.max(ph_flat, dim=1)  # each keypoint max & index
            valid_k = max_vals > 0.1  # (K,)

            ys = (max_ids // W).long()  # (K,)
            xs = (max_ids % W).long()  # (K,)

            # get each keypoint's pixel's face index
            face_ids = pix_to_face[ys, xs]
            valid_face = face_ids >= 0
            valid_all = valid_face#valid_k & valid_face

            # gather barycentric coords per keypoint
            bary_k = bary_coords[ys, xs]  # (K,3)

            # compute vertex coords for each face
            faces = mesh_p3d.faces_packed()
            verts = mesh_p3d.verts_packed()

            v0 = verts[faces[face_ids.clamp(min=0), 0]]  # (K,3)
            v1 = verts[faces[face_ids.clamp(min=0), 1]]
            v2 = verts[faces[face_ids.clamp(min=0), 2]]

            coords_3d = (
                    bary_k[:, 0:1] * v0 +
                    bary_k[:, 1:2] * v1 +
                    bary_k[:, 2:3] * v2
            )  # (K,3)

            # invalidate bad ones (max < 0.9 or invisible)
            coords_3d[~valid_all] = -1
            max_vals[~valid_all] = -1
            face_ids[~valid_all] = -1

            coords_np = coords_3d.cpu().numpy()  # (K, 3)
            valid_mask = ~(coords_np == -1).all(axis=1)
            kps = coords_np[valid_mask]

            w_geo = max_vals[valid_mask].cpu()
            w_vis = pred_vis_fg[valid_mask].cpu()
            w = w_vis#w_geo * w_vis
            cls = fg_idx.cpu().numpy()[valid_mask]
            for i, sem in enumerate(cls):
                sem_weights[sem].append(w[i])
                sem_xyz[sem].append(kps[i])

        pred_kps = []
        pred_ws = []
        for sem, kps in sem_xyz.items():
            ws = np.array(sem_weights[sem])
            for i, xyz in enumerate(kps):
                pred_ws.append(ws[i])
                pred_kps.append(xyz)
        pred_kps = np.array(pred_kps)

        final_kps = cluster_predicted_keypoints(pred_kps, pred_ws, min_cluster_size=15)  # pred_ws
        diff = final_kps[:, None, :] - points[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)  # (M,N)
        indices = np.argmin(dists, axis=1)
        final_kps = points[indices]

        dists = geo_dists[:, indices]
        eu_dists = cdist(gt_kps, final_kps, metric='euclidean')
        cd = get_cd_dist(eu_dists)

        mcd.append(cd)
        for i in range(11):
            key = i * 0.01
            if i == 0: key = 0.001
            iou = get_geo_iou(dists, key)
            miou[key].append(iou)
        print(mesh_name, cd)

    for i in range(11):
        key = i * 0.01
        if i == 0: key = 0.001
        print(np.mean(miou[key]))
    print(np.mean(mcd))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    inference_and_map_to_mesh(args.checkpoint)
