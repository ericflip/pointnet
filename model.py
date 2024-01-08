import torch
import torch.nn as nn


class PointNet(nn.Module):
    pass


class TNet(nn.Module):
    def __init__(self, in_features=3):
        super().__init__()
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, 64)  # we can also use conv1d
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 1024)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_features=in_features**2),
        )

    def mlp(self, x: torch.Tensor):
        x = self.apply_linear(x, self.linear1, self.batchnorm1)
        x = self.apply_linear(x, self.linear2, self.batchnorm2)
        x = self.apply_linear(x, self.linear3, self.batchnorm3)

        return x

    def apply_linear(
        self, x: torch.Tensor, linear: nn.Linear, batchnorm: nn.BatchNorm1d
    ):
        x = linear(x)  # (N, C, F) -> (N, C, n_F) n_F = new features, C = # of points
        x = x.permute(
            (0, 2, 1)
        )  # (N, C, n_F) -> (N, n_F, C) to apply batchnorm across n_F dimension
        x = batchnorm(x)  # (N, n_F, C) -> (N, n_F, C)
        x = x.permute((0, 2, 1))  # (N, n_F, C) -> (N, C, n_F)

        x = self.relu(x)

        return x

    def max_pool(self, x: torch.Tensor):
        x = x.max(dim=1)  # (N, C, F) -> (N, F) max pool across dim=1

        return x.values

    def forward(self, x: torch.Tensor):
        in_ = x  # (N, C, in_feature)
        N = x.shape[0]

        x = self.mlp(x)  # (N, C, in_feature) -> (N, C, 1024)
        x = self.max_pool(x)  # (N, C, 1024) -> (N, 1024)
        x = self.fc(x)  # (N, 1024) -> (N, in_feature**2)

        # reshape to (N, 3, 3)
        x = x.view(N, self.in_features, self.in_features)

        # identity matrix
        iden = torch.eye(self.in_features).repeat(N, 1, 1).to(x.device)

        # transformation matrix (N, in_feature, in_feature)
        x = x + iden

        # matrix multiply
        out = torch.bmm(in_, x)  # (N, C, in_feature) x (N, in_feature, in_feature)

        return out, x # (output, transformation matrix)
