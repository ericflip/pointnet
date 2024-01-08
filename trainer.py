import torch
from dataset import Model10NetDataset
from loss import CrossEntropyWithFeatureRegularization
from model import PointNet
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


class Model10NetPointNetTrainer:
    def __init__(
        self,
        model: PointNet,
        epochs: int,
        train_set: Model10NetDataset,
        test_set: Model10NetDataset,
        batch_size=32,
        lr=1e-3,
        weight=0.001,
        checkpoint: str = "./checkpoint",
        device="cpu",
    ):
        self.model = model.to(device)
        self.epochs = epochs
        self.train_set = train_set = train_set
        self.test_set = test_set
        self.train_loader = self.create_dataloader(train_set, batch_size=batch_size)
        self.test_loader = self.create_dataloader(
            test_set, batch_size=batch_size, shuffle=False
        )
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.loss = CrossEntropyWithFeatureRegularization(weight=weight)
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.device = device

    def create_dataloader(
        self, dataset: Model10NetDataset, batch_size: int = 8, shuffle=True
    ):
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
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )

    def train(self):
        model = self.model
        model.train()

        optimizer = self.optimizer

        losses = []

        for epoch in tqdm(range(self.epochs), position=0):
            pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch}]")
            # total_loss =
            for batch in pbar:
                X, Y = batch
                X, Y = X.to(self.device), Y.to(self.device)

                X = X.permute((0, 2, 1))  # (N, P, 3) -> (N, 3, P)

                pred, feature_transform = model(X)
                loss = self.loss(pred, Y, feature_transform)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())

        # save model weights

    def eval(self):
        pass
