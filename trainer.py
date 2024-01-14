import os
import pprint

import torch
import torch.optim.lr_scheduler as lr_scheduler
from dataset import Model10NetDataset
from evals import eval_pointnet
from loss import CrossEntropyWithFeatureRegularization
from model import PointNet
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)


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
        checkpoint_every=1,
        evaluate_every=10,
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
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.loss = CrossEntropyWithFeatureRegularization(weight=weight)
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.checkpoint_every = checkpoint_every
        self.evaluate_every = evaluate_every
        self.device = device

    def create_dataloader(
        self, dataset: Model10NetDataset, batch_size: int = 8, shuffle=True
    ):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def save_checkpoint(self, losses: list[int], path):
        weights = self.model.state_dict()
        checkpoint = {"losses": losses, "weights": weights}

        torch.save(checkpoint, path)

    def save_checkpoint_by_epoch(self, losses: list[float], epoch: int):
        path = os.path.join(self.checkpoint, f"epoch-{epoch}.pt")
        print("==CHECKPOINTING==")
        self.save_checkpoint(losses, path)

    def save_evals(self, path: str):
        evals = self.eval()
        print("==EVALS==")
        pp.pprint(evals)
        torch.save(evals, path)

    def save_evals_by_epoch(self, epoch: int):
        path = os.path.join(self.checkpoint, f"epoch-{epoch}-evals.pt")
        self.save_evals(path)

    def train(self):
        model = self.model
        optimizer = self.optimizer
        losses = []

        for epoch in range(self.epochs):
            model.train()
            pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch}]", leave=True)
            total_loss = 0

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
                total_loss += loss.item()

                pbar.set_postfix(loss=loss.item())

            epoch_loss = total_loss / len(self.train_loader)
            print(f"[Epoch {epoch}]: Loss={epoch_loss}")

            # update learning rate
            self.scheduler.step()

            # save model weights and losses every epoch
            if (epoch + 1) % self.checkpoint_every == 0:
                self.save_checkpoint_by_epoch(losses, epoch)

            if (epoch + 1) % self.evaluate_every == 0:
                self.save_evals_by_epoch(epoch)

        # save final model weights and losses
        self.save_checkpoint(losses, os.path.join(self.checkpoint, "final.pt"))
        self.save_evals(os.path.join(self.checkpoint, "final-evals.pt"))

    def eval(self):
        return eval_pointnet(self.model, self.test_set, self.device)
