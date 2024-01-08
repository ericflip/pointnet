import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet1 = TNet(in_features=3)

    def forward(self, x: torch.Tensor):
        pass


class TNet(nn.Module):
    """
    Transformation Network: Makes input invariant to geometric transformations (ie. rotations)
    """

    def __init__(self, in_features=3):
        super().__init__()
        self.in_features = in_features

        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
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

    def max_pool(self, x: torch.Tensor):
        x = x.max(dim=-1)  # (N, F, C) -> (N, F) max pool across dim=1

        return x.values

    def forward(self, x: torch.Tensor):
        in_ = x  # (N, in_feature, C), C = # of points
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

        # matrix multiply
        out = torch.bmm(trans, in_)  # (N, C, in_feature) x (N, in_feature, in_feature)

        return out, trans  # (output, transformation matrix)
