import json
import os

import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.io.pluggable_formats import PointcloudFormatInterpreter
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Pointclouds
from pytorch3d.structures import Meshes

io = IO()

KEYPOINTNET_DATASET_PATH = os.environ.get("KEYPOINT_DATASET_PATH", "keypointnet")
MESHES_PATH = os.path.join(KEYPOINTNET_DATASET_PATH, os.environ.get("KEYPOINT_DATASET_MESHES_PATH",
                                                                    "ShapeNetCore.v2.ply"))
PCDS_PATH = os.path.join(KEYPOINTNET_DATASET_PATH, os.environ.get("KEYPOINT_DATASET_PCDS_PATH",
                                                                  "pcds"))
KEYPOINTS_PATH = os.path.join(KEYPOINTNET_DATASET_PATH, "annotations")

def remove_unreferenced_vertices(mesh: Meshes) -> Meshes:
    """
    Remove vertices that are not referenced by any face, and reindex faces.
    Works for a single-mesh Meshes object (most common case).
    Returns a new Meshes on the same device/dtype.
    """
    # assert mesh.num_meshes() == 1, "This helper assumes a single mesh. Extend if you have batched meshes."
    device = mesh.device

    verts = mesh.verts_packed()   # (V, 3)
    faces = mesh.faces_packed()   # (F, 3)

    V = verts.shape[0]

    used = torch.zeros(V, dtype=torch.bool, device=device)
    used[faces.view(-1)] = True
    if used.all():
        return mesh

    new_vids = torch.nonzero(used, as_tuple=False).view(-1)
    new_V = new_vids.shape[0]

    idx_map = torch.full((V,), -1, dtype=torch.long, device=device)
    idx_map[new_vids] = torch.arange(new_V, dtype=torch.long, device=device)

    new_faces = idx_map[faces]

    new_verts = verts[new_vids]

    new_mesh = Meshes(verts=[new_verts], faces=[new_faces])

    return new_mesh

def load_mesh(mesh_file, use_texture=True):
    """
    Loads the mesh with the given ID.
    """
    mesh = io.load_mesh(mesh_file, include_textures=use_texture)
    if not use_texture:
        mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed()[None]) * 0.7)
    return mesh

def load_pcd(mesh_file):
    """
    Loads the point cloud for the given class and mesh ID.
    """
    pcd = io.load_pointcloud(mesh_file)
    return pcd


def load_keypoints():
    """
    Loads the keypoints from the annotations folder.
    """
    keypoints = {}
    for file in os.listdir(KEYPOINTS_PATH):
        with open(os.path.join(KEYPOINTS_PATH, file), "r") as f:
            labels = json.load(f)
            for l in labels:
                if l["class_id"] not in keypoints:
                    keypoints[l["class_id"]] = {}
                if l["model_id"] in keypoints[l["class_id"]]:
                    continue
                keypoints[l["class_id"]][l["model_id"]] = l["keypoints"]
    return keypoints


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=float)
    colors = np.array(data[:, -1], dtype=int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


class PCDKeypointNetInterpreter(PointcloudFormatInterpreter):
    def read(self, path, device, path_manager, **kwargs):
        pc, colors = naive_read_pcd(path)
        return Pointclouds(points=[torch.from_numpy(pc).to(device)],
                           features=[torch.from_numpy(colors).to(device)])


io.register_pointcloud_format(PCDKeypointNetInterpreter())