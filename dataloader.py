import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset import QADataset


class MDLoader(pl.LightningDataModule):
    def __init__(self, dataset_path, num_workers, model_name):
        super().__init__()
        self.test_ds = None
        self.train_ds = None
        self.val_ds = None
        self.pred_ds = None
        self.pre_val_ds = None
        self.pre_pred_ds = None
        self.pre_train_ds = None
        self.pre_test_ds = None
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.model_name = model_name

    def setup(self, stage: str):
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)


        self.pre_train_ds, self.pre_test_ds = random_split(dataset, [0.9, 0.1])
        self.pre_train_ds, self.pre_val_ds = random_split(self.pre_train_ds, [0.9, 0.1])

        self.train_ds = QADataset(self.model_name, self.pre_train_ds)
        self.val_ds = QADataset(self.model_name, self.pre_val_ds)
        self.test_ds = QADataset(self.model_name, self.pre_test_ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.pred_ds, batch_size=1, num_workers=self.num_workers)