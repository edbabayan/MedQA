from dataloader import MDLoader
from model import T5QA
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

torch.manual_seed(42)

logger = TensorBoardLogger('tb_logs', name='qa_model_v0')

dataloader = MDLoader(dataset_path='JSON_files/train.json', num_workers=2, model_name='t5-base')

model = T5QA('t5-base')

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    monitor="train_loss",
    mode="min"
)

trainer = Trainer(accelerator='gpu',  devices=-1, max_epochs=5, logger=logger,
                  accumulate_grad_batches=2, callbacks=[checkpoint_callback])

trainer.fit(model, dataloader)
trainer.validate(model, dataloader)
trainer.test(model, dataloader)
