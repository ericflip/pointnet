import torch
from dataset import Model10NetDataset
from model import PointNet
from torch.utils.data import DataLoader


class Model10NetPointNetTrainer:
    def __init__(
        self,
        model: PointNet,
        epochs: int,
        train_set: Model10NetDataset,
        test_set: Model10NetDataset,
        batch_size: int,
    ):
        self.model = model
        self.epochs = epochs
        self.train_set = train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size

    def create_dataloader(self, dataset: Model10NetDataset, batch_size: int = 8):
        def collate_fn(point_clouds: list[tuple[torch.Tensor, int]]):
            max_points = max([point_cloud.shape[0] for point_cloud, _ in point_clouds])

            classes = []
            padded_point_clouds = []

            for point_cloud, cls in point_clouds:
                classes.append(cls)

                # number of points
                P = point_cloud.shape[0]

                # points to pad
                pad = max_points - P
                pad_points = torch.zeros((pad, 3))

                # pad point cloud
                pad_point_cloud = torch.cat((point_cloud, pad_points), dim=0)
                padded_point_clouds.append(pad_point_cloud)

            padded_point_clouds = torch.stack(padded_point_clouds)
            classes = torch.tensor(classes)

            return padded_point_clouds, classes

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

    def train(self):
        pass

    def eval(self):
        pass
