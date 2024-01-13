import os

import torch
from torch.utils.data import Dataset


def create_classes_map(classes: list[str]):
    return {cls: i for i, cls in enumerate(classes)}


def parse_off(path: str):
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

        # Read faces
        faces = [
            list(map(int, file.readline().strip().split()))[1:] for _ in range(n_faces)
        ]  # ignoring the first number which is vertex count in face

        vertices = torch.tensor(vertices)
        faces = torch.tensor(faces)

        return vertices, faces


class Model10NetDataset(Dataset):
    def __init__(self, path: str, train=True):
        super().__init__()
        self.path = path
        self.train = train

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
        point_cloud = parse_off(object_path)

        return point_cloud, class_idx

    def __len__(self):
        return len(self.objects)


def area_of_faces(faces: torch.Tensor) -> torch.Tensor:
    """
    Calculates area of faces in a batch of point clouds using Heron's formula.
    """

    a, b, c = faces[:, 0, :], faces[:, 1, :], faces[:, 2, :]

    s1 = torch.linalg.norm(a - b, dim=1)
    s2 = torch.linalg.norm(a - c, dim=1)
    s3 = torch.linalg.norm(b - c, dim=1)

    s = (s1 + s2 + s3) / 2

    w = (s * (s - s1) * (s - s2) * (s - s3)).clamp(
        min=0
    )  # we have degenerate triangles so we clamp w to min=0
    area = torch.sqrt(w)

    return area


def sample_from_face(vertices: torch.Tensor, k: int):
    # Generate random numbers
    r = torch.rand((k))
    s = torch.rand((k))

    # Ensure the points are inside the triangle
    outside = r + s > 1
    r[outside] = 1 - r[outside]
    s[outside] = 1 - s[outside]

    # Compute the points
    A = vertices[0]
    B = vertices[1]
    C = vertices[2]

    P = A + r.unsqueeze(1) * (B - A) + s.unsqueeze(1) * (C - A)

    return P


class SamplePointCloudDataset(Dataset):
    def __init__(self, dataset: Model10NetDataset, k=1024):
        """k - # of samples"""
        super().__init__()
        self.dataset = dataset
        self.k = k

    @property
    def classes_map(self):
        return self.dataset.classes_map

    def __getitem__(self, index: int):
        point_cloud, class_idx = self.dataset[index]
        vertices, faces = point_cloud

        faces_array = faces.view((-1))
        points_on_faces = vertices[faces_array].view((-1, 3, 3))

        # normalize into unit sphere [-1, 1]
        x = points_on_faces[:, :, 0]
        points_on_faces[:, :, 0] = (x - x.mean()) / x.max()

        y = points_on_faces[:, :, 1]
        points_on_faces[:, :, 1] = (y - y.mean()) / y.max()

        z = points_on_faces[:, :, 2]
        points_on_faces[:, :, 2] = (z - z.mean()) / z.max()

        # calculate area of faces
        areas = area_of_faces(points_on_faces)

        # sample from mesh uniformly based on face area
        total_area = areas.sum()

        # proportions
        proportions = areas / total_area
        num_samples = proportions * self.k

        curr = 0
        sampled_points = []

        for i, num_sample in enumerate(num_samples):
            if curr == self.k:
                break

            num_points = int(num_sample.ceil().item())
            face_vertices = points_on_faces[i]
            samples = sample_from_face(face_vertices, k=num_points)

            if curr + num_points <= self.k:
                sampled_points.append(samples)
            elif curr + num_points > self.k:
                sampled_points.append(samples[: self.k - curr])

            curr += num_points

        sampled_points = torch.cat(sampled_points)

        return sampled_points, class_idx

    def __len__(self):
        return len(self.dataset)
