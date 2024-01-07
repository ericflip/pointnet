import os

import torch
from torch.utils.data import Dataset


def create_classes_map(classes: list[str]):
    return {cls: i for i, cls in enumerate(classes)}


def point_cloud_from_off_file(path: str) -> torch.Tensor:
    with open(path, "r") as file:
        if "OFF" != file.readline().strip():
            raise ValueError("Not a valid OFF file")

        # Read the counts of vertices, faces, and edges
        n_vertices, n_faces, _ = map(int, file.readline().strip().split(" "))

        # Read the vertex data
        vertices = [
            list(map(float, file.readline().strip().split(" ")))
            for _ in range(n_vertices)
        ]

        return torch.tensor(vertices)


class Model10NetDataset(Dataset):
    def __init__(self, path: str, train=True):
        super().__init__()
        self.path = path
        self.train = train

        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        classes_map = create_classes_map(classes)

        # list of tuples (path_to_object, class_idx)
        self.objects = []

        for cls, class_idx in classes_map.items():
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
        point_cloud = point_cloud_from_off_file(
            object_path
        )  # (N, 3) -> N = # of points

        return point_cloud, class_idx

    def __len__(self):
        return len(self.objects)
