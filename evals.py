import torch
from dataset import Model10NetDataset, SamplePointCloudDataset
from model import PointNet
from torch.utils.data import DataLoader


def eval_pointnet(model: PointNet, dataset: Model10NetDataset, device="cpu"):
    pointcloud_dataset = SamplePointCloudDataset(dataset)
    dataloader = DataLoader(pointcloud_dataset, batch_size=32)

    model = model.to(device)

    # set to eval mode
    model.eval()

    with torch.no_grad():
        for X, Y in dataloader:
            print(X, Y)
            break


if __name__ == "__main__":
    pass
