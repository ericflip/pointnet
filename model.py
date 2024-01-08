import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    """
    Params:
        - k: num classes
    """

    def __init__(self, k=10):
        super().__init__()
        self.input_transform = TNet(in_features=3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.feature_transform = TNet(in_features=64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.batchnorm1 = nn.BatchNorm1d(3)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.classifier = PointNetClassifier(k=k)

    def max_pool(self, x: torch.Tensor):
        return x.max(dim=-1)[0]  # (N, 1024, P) -> # (N, 1024)

    def forward(self, x: torch.Tensor):
        """
        Params:
            - x: (N, 3, P) tensor where N is batch size and P is number of points
        """

        # input transform
        input_transform = self.input_transform(x)  # (N, 3, P) -> (N, 3, 3)
        x = torch.bmm(input_transform, x)  # (N, 3, 3) x (N, 3, P) -> (N, 3, P)
        x = F.relu(self.batchnorm1(x))

        # mlp
        x = self.mlp1(x)

        # feature transform
        feature_transform = self.feature_transform(x)
        x = torch.bmm(feature_transform, x)  # (N, 64, 64) x (N, 64, P) -> (N, 64, P)
        x = F.relu(self.batchnorm2(x))

        # mlp
        x = self.mlp2(x)

        # max pooling
        x = self.max_pool(
            x
        )  # makes the input order-invariant -> max is a symmetric function

        # classifier
        x = self.classifier(x)

        return x, input_transform, feature_transform


class PointNetClassifier(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.classifer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k),
        )

    def forward(self, x: torch.Tensor):
        return self.classifer(x)  # (N, 1024) -> (N, k)


class TNet(nn.Module):
    """
    Transformation Network: Makes input invariant to geometric transformations (ie. rotations)
    """

    def __init__(self, in_features=3):
        super().__init__()

        self.in_features = in_features
        self.mlp = nn.Sequential(
            nn.Conv1d(in_features, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_features=in_features**2),
        )

        self.batchnorm = nn.BatchNorm1d(in_features)

    def max_pool(self, x: torch.Tensor):
        x = x.max(dim=-1)  # (N, F, C) -> (N, F) max pool across dim=1

        return x.values

    def forward(self, x: torch.Tensor):
        N = x.shape[0]

        x = self.mlp(x)  # (N, in_feature, C) -> (N, 1024, C)
        x = self.max_pool(x)  # (N, 1024, C) -> (N, 1024)
        x = self.fc(x)  # (N, 1024) -> (N, in_feature**2)

        # reshape to (N, 3, 3)
        x = x.view(N, self.in_features, self.in_features)

        # identity matrix
        iden = torch.eye(self.in_features).repeat(N, 1, 1).to(x.device)

        # transformation matrix (N, in_feature, in_feature)
        trans = x + iden

        return trans
