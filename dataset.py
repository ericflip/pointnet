import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset


def create_classes_map(classes: list[str]):
    return {cls: i for i, cls in enumerate(classes)}


class Model10NetDataset(Dataset):
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

        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        self.classes_map = create_classes_map(classes)

        # list of tuples (path_to_object, class_idx)
        self.objects = []

        for cls, class_idx in self.classes_map.items():
            split = "train" if train else "test"
            class_path = os.path.join(path, cls, split)
            objects = [
                (os.path.join(class_path, f), class_idx)
                for f in os.listdir(class_path)
                if f != ".DS_Store"
            ]

            self.objects += objects

    def __getitem__(self, index: int):
        object_path, class_idx = self.objects[index]
        mesh = trimesh.load(object_path, file_type="off")

        sampled_points = trimesh.sample.sample_surface(mesh, count=self.k)[0]
        sampled_points = np.array(sampled_points)

        # normalize to unit sphere
        mean = sampled_points.mean(axis=0)
        sampled_points -= mean
        sampled_points /= np.abs(sampled_points).max(axis=0)

        if self.data_augmentation:
            # rotate
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            sampled_points[:, [0, 2]] = sampled_points[:, [0, 2]].dot(rotation_matrix)

            # add gaussian noise with mean 0 and std 0.02
            jitter = np.random.normal(loc=0, scale=0.02, size=sampled_points.shape)
            sampled_points += jitter

        sampled_points = torch.from_numpy(sampled_points)

        return sampled_points, class_idx

    def __len__(self):
        return len(self.objects)
