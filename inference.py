import argparse

import torch
import torch.nn.functional as F
from dataset import Model10NetDataset
from model import PointNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with pointnet")
    parser.add_argument("--obj-path", type=str, help="Path to .off file")
    args = parser.parse_args()
    file_path = args.obj_path

    # parse .off file
    points = Model10NetDataset.load_and_sample_point_cloud(file_path)

    model = PointNet()

    points = points.permute((1, 0)).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred, _ = model(points)
        probabilities = F.softmax(pred, dim=-1)

        score, class_idx = probabilities[0].max(dim=-1)
        class_name = Model10NetDataset.CLASSES[class_idx]

        print(f"Predicted Class: {class_name} | Score: {score}")
