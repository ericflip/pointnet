import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset


def create_classes_map(classes: list[str]):
    return {cls: i for i, cls in enumerate(classes)}


CLASSES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

CLASSES_MAP = {class_: i for i, class_ in enumerate(CLASSES)}


class Model10NetDataset(Dataset):
    CLASSES = CLASSES
    CLASSES_MAP = CLASSES_MAP

    def __init__(self, path: str, k=1024, train=True, data_augmentation=False):
        """
        Params:
            - path: path of dataset
            - k: number of points to sample from pointcloud
            - train: split
            - data_augmentation: rotate pointcloud and add gaussian jitter
        """
        super().__init__()
        self.path = path
        self.k = k
        self.train = train
        self.data_augmentation = data_augmentation

        # list of tuples (path_to_object, class_idx)
        self.objects = []

        for cls, class_idx in self.CLASSES_MAP.items():
            split = "train" if train else "test"
            class_path = os.path.join(path, cls, split)
            objects = [
                (os.path.join(class_path, f), class_idx)
                for f in os.listdir(class_path)
                if f != ".DS_Store"
            ]

            self.objects += objects

    @staticmethod
    def load_and_sample_point_cloud(path: str, k=1024, normalize=True) -> torch.Tensor:
        mesh = trimesh.load(path, file_type="off")

        sampled_points = trimesh.sample.sample_surface(mesh, count=k)[0]
        sampled_points = torch.from_numpy(sampled_points).to(torch.float32)

        if normalize:
            sampled_points = Model10NetDataset.normalize_point_cloud(sampled_points)

        return sampled_points

    @staticmethod
    def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
        """Normalize to unit sphere"""

        mean = points.mean(axis=0)
        points -= mean
        points /= points.abs().max(axis=0)[0]

        return points

    def apply_augmentation(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to points and add gaussian jitter (mean=0, std=0.02)"""

        # rotate
        theta = torch.rand(1) * 2 * np.pi
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        )

        points[:, [0, 2]] = torch.mm(points[:, [0, 2]], rotation_matrix)

        # add gaussian noise with mean 0 and std 0.02
        jitter = torch.randn(points.shape) * 0.02
        points += jitter

        return points

    def __getitem__(self, index: int):
        object_path, class_idx = self.objects[index]
        sampled_points = Model10NetDataset.load_and_sample_point_cloud(object_path)

        if self.data_augmentation:
            sampled_points = self.apply_augmentation(sampled_points)

        return sampled_points, class_idx

    def __len__(self):
        return len(self.objects)
