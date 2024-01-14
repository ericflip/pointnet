import numpy as np
import torch
import trimesh


def load_and_sample_point_cloud(path: str, k=1024) -> torch.Tensor:
    mesh = trimesh.load(path, file_type="off")

    sampled_points = trimesh.sample.sample_surface(mesh, count=k)[0]
    sampled_points = torch.from_numpy(sampled_points).to(torch.float32)

    return sampled_points


def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """Normalize to unit sphere"""

    mean = points.mean(axis=0)
    points -= mean
    points /= points.abs().max(axis=0)[0]

    return points
