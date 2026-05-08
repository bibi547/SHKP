import os
import torch
import torchvision
import json
import numpy as np
from utils.load_utils import load_mesh, naive_read_pcd, remove_unreferenced_vertices
from utils.rendering import setup_renderer, sample_view_points
from pytorch3d.renderer import look_at_rotation, FoVPerspectiveCameras, PointLights
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


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
    root_dir = "./datasets/KeypointNet"

    num_view = 62
    img_size = 256

    class_name = 'airplane'
    render_dist = 1.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    anno_file = './datasets/KeypointNet/annotations/all.json'
    annots = json.load(open(anno_file))
    annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[class_name]]
    keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])

    pcd_root = os.path.join(root_dir, 'pcds', NAMES2ID[class_name])
    mesh_root = os.path.join(root_dir, 'ShapeNetCore.v2.ply', NAMES2ID[class_name])
    write_root = os.path.join(root_dir, 'images', NAMES2ID[class_name])

    renderer = setup_renderer(device)

    pcd_files = os.listdir(pcd_root)
    for j, f in enumerate(pcd_files):
        # if j < 400:continue

        filename = os.path.splitext(f)[0]
        pcd_file = os.path.join(pcd_root, filename + '.pcd')
        mesh_file = os.path.join(mesh_root, filename + '.ply')
        write_path = os.path.join(write_root, filename)
        os.makedirs(write_path, exist_ok=True)

        mesh = load_mesh(mesh_file).to(device)
        faces = mesh.faces_packed()
        verts = mesh.verts_packed()
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
        keep = areas > 1e-8
        mesh = Meshes(verts=[verts], faces=[faces[keep]])
        mesh = remove_unreferenced_vertices(mesh)
        mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed()[None]) * 0.7)

        points = naive_read_pcd(pcd_file)[0]
        pcds = torch.tensor(points, dtype=torch.float32, device=device)
        kps = keypoints[filename]
        kps_xyz = []
        kps_class = []
        for i, kp in enumerate(kps):
            # kp_idx = kp[0]
            kp_xyz = pcds[kp[0]]
            kp_class = kp[1]
            kps_xyz.append(kp_xyz)
            kps_class.append(kp_class)
        kps_xyz = torch.stack(kps_xyz, dim=0).to(device)
        kps_class = torch.tensor(kps_class, device=device)
        verts = mesh.verts_packed()  # (V, 3)
        K = kps_xyz.shape[0]
        dists = torch.norm(verts[None, :, :] - kps_xyz[:, None, :], dim=2)  # (K, V)
        kp_vids = torch.argmin(dists, dim=1)  # (K,)
        kp_vids = kp_vids.cpu().numpy()

        verts = mesh.verts_packed().cpu().numpy()  # (V, 3)
        faces = mesh.faces_packed().cpu().numpy()  # (F, 3)
        # dist = pp3d.compute_distance_multisource(verts, faces, kp_vids)

        res = renderer.rasterizer.raster_settings.image_size
        views = sample_view_points(render_dist, 5)

        num_views = len(views)
        # camera transform
        R = look_at_rotation(views, device=device)
        T = torch.tensor([0, 0, render_dist], device=device).repeat(len(views), 1)

        for i in range(num_views):
            view = views[i][np.newaxis, :]
            # 1. Render the mesh
            camera = FoVPerspectiveCameras(R=R[i].unsqueeze(0), T=T[i].unsqueeze(0), device=device)
            light = PointLights(ambient_color=((0.5, 0.5, 0.5),), location=view, device=device)

            with torch.no_grad():
                images, fragments = renderer(mesh, cameras=camera, lights=light)
                images = images[..., :3]

            pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H, W)
            mask = (pix_to_face >= 0).float()

            H, W = pix_to_face.shape
            valid = pix_to_face >= 0
            valid_faces = pix_to_face[valid]  # (N_valid,)
            bary_coords = fragments.bary_coords[0, ..., 0, :]  # shape: (H, W, 3)
            bary_valid = bary_coords[valid]  # (N_valid, 3)
            faces_idx = mesh.faces_packed()[valid_faces]  # (N_valid, 3)
            d0 = dists[:, faces_idx[:, 0].cpu()].T  # (N_valid, K)
            d1 = dists[:, faces_idx[:, 1].cpu()].T
            d2 = dists[:, faces_idx[:, 2].cpu()].T

            b0 = bary_valid[:, 0:1].expand(-1, K)
            b1 = bary_valid[:, 1:2].expand(-1, K)
            b2 = bary_valid[:, 2:3].expand(-1, K)

            dist_2d = torch.zeros((H, W, K), device=device, dtype=torch.float32)
            dist_2d[valid] = b0 * d0 + b1 * d1 + b2 * d2  # (N_valid, K)
            write_file_img = os.path.join(write_path, f'view_{i}.png')
            write_file_mask = os.path.join(write_path, f'view_{i}_mask.png')
            write_file_dist = os.path.join(write_path, f'view_{i}_dist.pt')
            write_file_coord = os.path.join(write_path, f'view_{i}_coord.pt')
            img = images[0].cpu().clamp(0, 1).permute(2, 0, 1)  # (3,H,W)
            torchvision.utils.save_image(img, write_file_img)
            torchvision.utils.save_image(mask.unsqueeze(0).cpu(), write_file_mask)  # (1,H,W)
            torch.save(dist_2d.cpu(), write_file_dist)
            torch.save(bary_coords, write_file_coord)

            print(filename + 'view' + str(i))


















