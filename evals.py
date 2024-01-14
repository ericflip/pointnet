import torch
from dataset import Model10NetDataset
from model import PointNet
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_pointnet(model: PointNet, dataset: Model10NetDataset, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=32)

    model = model.to(device)

    # set to eval mode
    model.eval()
    total_correct = 0

    # confusion matrix
    num_classes = len(dataset.CLASSES)

    # axis=0 = actual, axis=1 = predicted
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int)

    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc="[EVALS]"):
            X, Y = X.to(device), Y.to(device)
            X = X.permute((0, 2, 1))

            pred, feature_transform = model(X)
            pred_classes = pred.max(dim=1)[1]

            num_correct = (pred_classes == Y).sum()
            total_correct += num_correct.item()

            for i, j in zip(Y.to("cpu"), pred_classes.to("cpu")):
                confusion_matrix[i][j] += 1

    accuracy = total_correct / len(dataset)

    diagonal = torch.diag(confusion_matrix)
    predicted_totals = confusion_matrix.sum(dim=0)
    class_accuracy = (diagonal / predicted_totals).nan_to_num(nan=0)

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_accuracy": class_accuracy.tolist(),
    }
