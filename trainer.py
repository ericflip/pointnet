from dataset import Model10NetDataset
from model import PointNet


class Model10NetPointNetTrainer:
    def __init__(
        self,
        model: PointNet,
        epochs: int,
        train_set: Model10NetDataset,
        test_set: Model10NetDataset,
        batch_size: int,
    ):
        self.model = model
        self.epochs = epochs
        self.train_set = train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size

    def train(self):
        pass

    def eval(self):
        pass
