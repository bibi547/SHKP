import torch
import numpy as np

# -------------------- SH basis (real SH L<=2) --------------------
def sh_basis_L2(dirs: torch.Tensor):
    """
    dirs: (N,3) normalized (torch)
    returns: (N,9)
    """
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    Y = torch.zeros((dirs.shape[0], 9), dtype=dirs.dtype, device=dirs.device)
    Y[:, 0] = 0.282095                           # l=0
    Y[:, 1] = 0.488603 * y                       # l=1
    Y[:, 2] = 0.488603 * z
    Y[:, 3] = 0.488603 * x
    Y[:, 4] = 1.092548 * x * y                   # l=2
    Y[:, 5] = 1.092548 * y * z
    Y[:, 6] = 0.315392 * (3 * z * z - 1.0)
    Y[:, 7] = 1.092548 * x * z
    Y[:, 8] = 0.546274 * (x * x - y * y)
    return Y


# -------------------- solve per-model SH weights --------------------
def solve_sh_weights_for_model(model_records, num_sem, blend=(1/2,1/2,1), lam=1e-3, device='cpu'):
    """
    model_records: list of tuples per view: (sem_ids (K,), vis_stats (K,3))
    view_dirs_np: (V,3) numpy array in same order as model_records
    returns: sh_weights (num_sem, 9) numpy
    """
    records = []  # (sem, dir_np, peak, hv, area)
    for (sem_ids, vis_stats), dir_np in model_records:
        sems = np.array(sem_ids).reshape(-1)
        vs = vis_stats.numpy() if isinstance(vis_stats, torch.Tensor) else np.array(vis_stats)
        for i, s in enumerate(sems):
            if int(s) == 0:
                continue
            records.append((int(s), dir_np.copy(), float(vs[i, 0]), float(vs[i, 1]), float(vs[i, 2])))

    if len(records) == 0:
        return np.zeros((num_sem, 9), dtype=np.float32)

    sems = np.array([r[0] for r in records], dtype=np.int64)
    dirs = np.array([r[1] for r in records], dtype=np.float32)
    peaks = np.array([r[2] for r in records], dtype=np.float32)
    hvs = np.array([r[3] for r in records], dtype=np.float32)
    areas = np.array([r[4] for r in records], dtype=np.float32)

    for c in range(num_sem):
        idx = (sems == c)
        if idx.sum() == 0:
            continue
        p99_peak = np.percentile(peaks[idx], 99) if idx.sum() > 1 else peaks[idx].max()
        p99_hv = np.percentile(hvs[idx], 99) if idx.sum() > 1 else hvs[idx].max()
        peaks[idx] = peaks[idx] / (p99_peak + 1e-6)
        hvs[idx] = hvs[idx] / (p99_hv + 1e-6)

    w1, w2, w3 = blend
    y = w1 * peaks + w2 * hvs + w3 * areas
    y = np.clip(y, 0.0, 1.0)

    # compute SH basis and solve ridge per-sem
    dirs_t = torch.from_numpy(dirs).float().to(device)
    Y_all = sh_basis_L2(dirs_t).cpu().numpy()   # (N,9)

    sh_weights = np.zeros((num_sem, 9), dtype=np.float32)
    for c in range(num_sem):
        idx = (sems == c)
        if idx.sum() < 3:
            continue
        Yc = Y_all[idx]
        yc = y[idx]
        A = Yc.T @ Yc + lam * np.eye(9)
        b = Yc.T @ yc
        w = np.linalg.solve(A, b)
        sh_weights[c] = w
    return sh_weights

# -------------------- compute vis_stats from GT heatmaps --------------------
def compute_vis_stats_from_heatmaps(gt_heatmaps, mask, area_thresh=0.9):
    """
    gt_heatmaps: (K, H, W) torch
    mask: (H, W) boolean torch
    returns: vis_stats (K,3): peak, hv, area_fraction
    """
    valid_pixels = mask.float().sum().clamp_min(1.0)
    peak = gt_heatmaps.amax(dim=[1,2])
    hvr = gt_heatmaps.sum(dim=[1,2]) / valid_pixels
    area = (gt_heatmaps > area_thresh).float().sum(dim=[1,2]) / valid_pixels

    return torch.stack([peak, hvr, area], dim=1)

# -------------------- compute vis_stats from GT heatmaps --------------------
def compute_vis_stats_from_heatmaps_original(gt_heatmaps, mask, area_thresh=0.9):
    """
    gt_heatmaps: (K, H, W) torch
    mask: (H, W) boolean torch
    returns: vis_stats (K,3): peak, hvr, area_fraction
    """
    valid_pixels = mask.float().sum().clamp_min(1.0)
    peak = gt_heatmaps.amax(dim=[1,2])
    hvr = gt_heatmaps.sum(dim=[1,2]) / valid_pixels
    area = (gt_heatmaps > area_thresh).float().sum(dim=[1,2]) / valid_pixels

    scale = 20.0
    hvr_out = torch.clamp(hvr * scale, 0, 1)  # 0-1
    area_out = torch.clamp(area * scale, 0, 1)

    y = peak / 2 + hvr_out + area_out * 10

    return y#torch.stack([peak, hvr_out, area_out], dim=1)
