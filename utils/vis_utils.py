import torch
import hdbscan
import numpy as np
from pytorch3d.renderer import look_at_rotation, FoVPerspectiveCameras, PointLights, RasterizationSettings, MeshRasterizer
from pytorch3d.structures import Meshes


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

def cluster_predicted_keypoints(pred_kps, pred_weights=None,
                                min_cluster_size=10, min_samples=None):

    if pred_weights is None:
        weights = None
    else:
        weights = np.asarray(pred_weights, dtype=np.float32)
        weights = np.clip(weights, 1e-3, None)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )

    clusterer.fit(pred_kps)

    labels = clusterer.labels_          # (N,)
    unique_labels = [l for l in set(labels) if l != -1]

    centers = []
    for c in unique_labels:
        idx = np.where(labels == c)[0]
        cluster_pts = pred_kps[idx]

        if pred_weights is None:
            center = cluster_pts.mean(axis=0)
        else:
            cluster_w = weights[idx]
            center = np.average(cluster_pts, axis=0, weights=cluster_w)

        centers.append(center)

    centers = np.array(centers)

    return centers

