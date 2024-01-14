import argparse
import os

import torch
from dataset import Model10NetDataset
from model import PointNet
from torch.utils.data import Subset
from trainer import Model10NetPointNetTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pointnet")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument(
        "--checkpoint-every", type=int, default=1, help="How many epochs to save model"
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=1,
        help="How many epochs to evaluate model",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="./checkpoint", help="Path to checkpoint"
    )

    # parse args
    args = parser.parse_args()
    dataset_path = args.dataset_path
    batch_size = args.batch
    lr = args.lr
    epochs = args.epochs
    checkpoint_path = args.checkpoint_path
    checkpoint_every = args.checkpoint_every
    evaluate_every = args.evaluate_every

    # create checkpoint dir
    os.makedirs(checkpoint_path, exist_ok=True)

    # reproducibility
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PointNet(k=10)
    train_set = Model10NetDataset(dataset_path, train=True, data_augmentation=True)
    # train_set = Subset(train_set, range(10))
    test_set = Model10NetDataset(dataset_path, train=False, data_augmentation=True)

    # initialize trainer
    trainer = Model10NetPointNetTrainer(
        model=model,
        epochs=epochs,
        train_set=train_set,
        test_set=test_set,
        batch_size=batch_size,
        lr=lr,
        checkpoint=checkpoint_path,
        checkpoint_every=checkpoint_every,
        evaluate_every=evaluate_every,
        device=device,
    )

    trainer.train()
