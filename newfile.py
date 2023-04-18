import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split



class QADataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=512)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context = self.data[idx]["Context"]
        qa_pair = self.data[idx]["QA"][0]
        question = qa_pair["Question"]
        answer = qa_pair["Answer"]

        input_text = question + " [SEP] " + context

        tokens = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        target_tokens = self.tokenizer.encode_plus(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        target_input_ids = target_tokens["input_ids"]

        return {
            "input_ids": input_ids.flatten(),
            "attention_mask": attention_mask.flatten(),
            "decoder_input_ids": target_input_ids.flatten(),
            'answer': answer
        }


class MDLoader(pl.LightningDataModule):
    def __init__(self, train_path, test_path, num_workers, model_name):
        super().__init__()
        self.test_ds = None
        self.train_ds = None
        self.val_ds = None
        self.pre_val_ds = None
        self.pre_train_ds = None
        self.pre_test_ds = None
        self.train = train_path
        self.test = test_path
        self.num_workers = num_workers
        self.model_name = model_name

    def setup(self, stage: str):
        with open(self.train, 'r') as f:
            train_data = json.load(f)
        with open(self.test, 'r') as f:
            self.pre_test_ds = json.load(f)

        self.pre_train_ds, self.pre_val_ds = random_split(train_data, [0.9, 0.1])

        self.train_ds = QADataset(self.model_name, self.pre_train_ds)
        self.val_ds = QADataset(self.model_name, self.pre_val_ds)
        self.test_ds = QADataset(self.model_name, self.pre_test_ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers)


class T5QA(pl.LightningModule):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask):
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(1, -1)
        attention_mask = batch["attention_mask"].view(1, -1)
        y_true = batch["decoder_input_ids"].flatten()
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)[1]
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y_true)
        self.log_dict({'train_loss': loss}, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(1, -1)
        attention_mask = batch["attention_mask"].view(1, -1)
        y_true = batch["decoder_input_ids"].flatten()
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)[1]
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y_true)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(1, -1)
        attention_mask = batch["attention_mask"].view(1, -1)
        y_true = batch["decoder_input_ids"].flatten()
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)[1]
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y_true)
        self.log('test_loss', loss)
        return loss

    def predict(self, input_text):
        encoded = self.tokenizer(input_text,
                                 add_special_tokens=True,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512,
                                padding="max_length",)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)[1]
        predicted_token_ids = output.argmax(dim=-1)
        predicted_text = self.tokenizer.decode(predicted_token_ids.squeeze().tolist(), skip_special_tokens=True)
        return predicted_text


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)



dataloader = MDLoader(train_path='JSON_files/train.json', test_path='JSON_files/test.json',
                      num_workers=4, model_name='t5-base')

model = T5QA("t5-base")

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    monitor="train_loss",
    mode="min"
)

trainer = Trainer(accelerator='gpu', precision=16, devices=-1, max_epochs=1, accumulate_grad_batches=2, callbacks=[checkpoint_callback])

trainer.fit(model, dataloader)
print(trainer.callback_metrics['val_loss'])
trainer.save_checkpoint("t5qa.ckpt")
trainer.validate(model, dataloader)
trainer.test(model, dataloader)