import torch
import torch.nn as nn


class CrossEntropyWithFeatureRegularization(nn.Module):
    def __init__(self, weight=0.001):
        super().__init__()
        self.weight = weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self, pred: torch.Tensor, classes: torch.Tensor, feature_transform: torch.Tensor
    ) -> torch.Tensor:
        M, N = feature_transform.shape[1], feature_transform.shape[1]

        # make sure feature transform is a square matrix
        assert M == N

        cross_entropy_loss = self.cross_entropy(pred, classes)

        # compute regularization loss
        identity = torch.eye(N)
        identity_minus_AAT = identity - torch.bmm(
            feature_transform, feature_transform.transpose(1, 2)
        )

        loss_reg = (
            self.weight * identity_minus_AAT.square().sum()
        )  # forces feature_transform to be an orthogonal matrix

        return cross_entropy_loss + loss_reg
