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
from pytorch3d.renderer import look_at_rotation, FoVPerspectiveCameras, PointLights, RasterizationSettings, MeshRasterizer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from datasets.utils import geodesic_heatmaps
from utils.sh_util import compute_vis_stats_from_heatmaps, solve_sh_weights_for_model, sh_basis_L2



def raster_fragments_for_view(mesh: Meshes, renderer, views, render_dist, view_idx, device):
    """
    Render mesh for the given view index and return pix_to_face (H,W), bary_coords (H,W,3) and fragments object.
    """
    # Compute camera transforms
    R = look_at_rotation(views, device=device)  # (N_views, 3, 3)
    T = torch.tensor([0, 0, render_dist], device=device).repeat(len(views), 1)

    # Create camera for the single view
    cam = FoVPerspectiveCameras(R=R[view_idx].unsqueeze(0), T=T[view_idx].unsqueeze(0), device=device)
    light = PointLights(ambient_color=((0.5, 0.5, 0.5),), location=views[view_idx][None, :], device=device)

    with torch.no_grad():
        images, fragments = renderer(mesh, cameras=cam, lights=light)
        pix_to_face = fragments.pix_to_face[0, ..., 0]   # (H, W)
        bary_coords = fragments.bary_coords[0, ..., 0, :]  # (H, W, 3)
    return pix_to_face, bary_coords, fragments


def sample_heatmap_at_vertices(mesh_p3d, heatmap, views, view_idx, render_dist, device):
    """
    Sample heatmap (H,W) at mesh vertices for the given view.
    Only keeps vertices that are:
        1. visible in the rasterization
        2. heatmap value > 0.95
    Invisible vertices or low heat vertices get value -1.

    Returns:
        (V,) tensor
    """

    if heatmap.dim() == 2:
        H, W = heatmap.shape
        heat_t = heatmap.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    else:
        raise ValueError("heatmap must be (H,W)")

    R = look_at_rotation(views, device=device)
    T = torch.tensor([0, 0, render_dist], device=device).repeat(len(views), 1)

    cam = FoVPerspectiveCameras(
        R=R[view_idx].unsqueeze(0),
        T=T[view_idx].unsqueeze(0),
        device=device
    )

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    rasterizer = MeshRasterizer(cameras=cam, raster_settings=raster_settings)
    fragments = rasterizer(mesh_p3d)
    pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H,W)

    visible_faces = pix_to_face.unique()
    visible_faces = visible_faces[visible_faces >= 0]

    faces = mesh_p3d.faces_packed()  # (F,3)
    visible_verts = torch.unique(faces[visible_faces].reshape(-1))

    verts = mesh_p3d.verts_packed().to(device)  # (V,3)
    verts_screen = cam.transform_points_screen(
        verts.unsqueeze(0), image_size=(H, W)
    )[0, :, :2]

    xs = verts_screen[:, 0]
    ys = verts_screen[:, 1]

    # normalize → [-1,1]
    x_norm = (xs / (W - 1)) * 2 - 1
    y_norm = (ys / (H - 1)) * 2 - 1
    y_norm = -y_norm  # flip y

    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, 1, -1, 2)

    sampled = F.grid_sample(
        heat_t, grid, mode='bilinear', align_corners=True
    ).view(-1)  # (V,)

    V = verts.shape[0]
    out = torch.full((V,), -1.0, device=device)

    heat_mask = sampled > 0.8
    vis_mask = torch.zeros(V, dtype=torch.bool, device=device)
    vis_mask[visible_verts] = True

    final_mask = heat_mask & vis_mask

    out[vis_mask] = sampled[vis_mask]
    return out


def inference_and_map_to_mesh(args,
                              test_file='./datasets/KeypointNet/splits/test.txt',
                              root_dir='./datasets/KeypointNet',):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_id = NAMES2ID[args.class_name]

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

    render_dist = 1.1
    N_VIEWS = 62
    renderer = setup_renderer(device)
    views = sample_view_points(render_dist, 5)  # (N_VIEWS, 3)
    view_dirs = views / np.linalg.norm(views, axis=1, keepdims=True)
    view_dirs_t = torch.from_numpy(view_dirs.astype(np.float32)).to(device)

    transform_mask = T.Compose([T.ToTensor()])

    for mesh_name, indices in tqdm.tqdm(model_to_indices.items(), desc="models"):
        # view_indices = sorted(model_to_indices[mesh_name])  # these correspond to dataset indices (per view)
        print("Processing:", mesh_name)
        if mesh_name not in model_to_indices:
            print("  no views found in dataset for", mesh_name)
            continue

        # load mesh (use the same loading as in preprocess)
        mesh_file = os.path.join(root_dir, 'ShapeNetCore.v2.ply', class_id, mesh_name + '.ply')
        if not os.path.exists(mesh_file):
            print("  mesh not found:", mesh_file)
            continue

        # 1) collect model-level GT vis_stats per view for SH solve
        model_records = []  # list of tuples (sem_ids_tensor, vis_stats_tensor, view_dir_np)
        for idx in indices:
            sample = dataset.samples[idx]
            kp_classes, img_file, mask_file, dino_file, dist_file = sample[:5]
            # infer view id
            basename = os.path.basename(img_file)
            try:
                view_id = int(basename.split('_')[1].split('.')[0])
            except:
                view_id = indices.index(idx)
            # load mask and dist -> GT heatmaps
            mask = Image.open(mask_file).convert("L")
            mask_t = T.ToTensor()(mask)[0] > 0.5
            dist = torch.load(dist_file)  # (H,W,K)
            dist = dist.permute(2, 0, 1).float()
            gt_heatmaps = geodesic_heatmaps(dist, args)
            gt_heatmaps = torch.from_numpy(gt_heatmaps).float() * mask_t.unsqueeze(0).float()
            # compute vis_stats (K,3)
            vis_stats = compute_vis_stats_from_heatmaps(gt_heatmaps, mask_t, area_thresh=0.9)  # (K,3)
            sem_ids = np.array(kp_classes).astype(np.int64)  # kp_classes is array-like length K
            model_records.append(((sem_ids, vis_stats), view_dirs[view_id]))
        # 2) solve per-model SH weights (num_sem x 9)
        num_sem = dataset.nclasses
        sh_weights_model = solve_sh_weights_for_model(
            model_records=model_records,
            num_sem=num_sem,
            blend=(1/2,1/2,1),
            lam=1e-3,
            device=device
        )
        sh_weights_model = np.array(sh_weights_model, dtype=np.float32)  # (num_sem,9)
        sh_weights_t = torch.from_numpy(sh_weights_model).float().to(device)

        # build pytorch3d Meshes for rendering
        mesh_p3d = load_mesh(mesh_file).to(device)  # returns pytorch3d.Meshes
        # ensure cleaned (same preprocessing)
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

        for idx in indices:
            kp_classes, img_file, mask_file, dino_file, dist_file = dataset.samples[idx]

            # infer view id from filename "view_{i}.png"
            basename = os.path.basename(img_file)
            view_id = int(basename.split('_')[1].split('.')[0])

            # load image and gt heatmaps
            kp_classes = np.array(kp_classes)

            # sh weight
            Yv = sh_basis_L2(view_dirs_t[view_id:view_id + 1, :]).squeeze(0)  # (9,)
            w_list = []
            # kp_classes = kp_classes[valid_mask]
            for sem in kp_classes:
                if sem == 0 or sem < 0 or sem >= num_sem:
                    w_list.append(0.0)
                else:
                    w_raw = (Yv * sh_weights_t[sem]).sum()
                    # w = torch.sigmoid(w_raw)  # scalar 0..1
                    w_list.append(float(w_raw.item()))
            w_sh = np.array(w_list, dtype=np.float32)

            folder = os.path.dirname(img_file)
            write_sh_file = os.path.join(folder, f"view_{view_id}_sh.npy")
            np.save(write_sh_file, w_sh)
            print("Saved SH weights to:", write_sh_file)




if __name__ == "__main__":
    class Args(object):
        def __init__(self):
            self.anno_dir = './datasets/KeypointNet/annotations/all.json'
            self.class_name = 'airplane'
            self.split_root = './datasets/KeypointNet/splits'
            self.train_file = 'train.txt'
            self.test_file = 'test.txt'
            self.img_root = './datasets/KeypointNet/images'
            self.decay_factor = 0.02
    inference_and_map_to_mesh(Args())
