import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AutoTokenizer
from rouge import Rouge


class T5QA(pl.LightningModule):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.rouge = Rouge()

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_input_ids)
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(1, -1)
        attention_mask = batch["attention_mask"].view(1, -1)
        decoder_input_ids = batch["decoder_input_ids"].view(1, -1)
        answer = batch['answer'][0]
        output = self.forward(input_ids, attention_mask, decoder_input_ids)
        loss = output.loss
        output_ids = output.logits.argmax(dim=-1).squeeze()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        score = self.rouge.get_scores(output_text, answer)[0]['rouge-2']['f']
        score = torch.tensor(score)
        self.log_dict({"train_loss": loss, 'train_f1': score}, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(1, -1)
        attention_mask = batch["attention_mask"].view(1, -1)
        decoder_input_ids = batch["decoder_input_ids"].view(1, -1)
        answer = batch['answer'][0]
        output = self.forward(input_ids, attention_mask, decoder_input_ids)
        loss = output.loss
        output_ids = output.logits.argmax(dim=-1).squeeze()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        score = self.rouge.get_scores(output_text, answer)[0]['rouge-2']['f']
        score = torch.tensor(score)
        self.log_dict({"val_loss": loss, 'val_f1': score}, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(1, -1)
        attention_mask = batch["attention_mask"].view(1, -1)
        decoder_input_ids = batch["decoder_input_ids"].view(1, -1)
        answer = batch['answer'][0]
        output = self.forward(input_ids, attention_mask, decoder_input_ids)
        loss = output.loss
        output_ids = output.logits.argmax(dim=-1).squeeze()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        score = self.rouge.get_scores(output_text, answer)[0]['rouge-2']['f']
        score = torch.tensor(score)
        self.log_dict({"test_loss": loss, 'test_f1': score}, on_step=True)
        return loss

    def predict_step(self, input_ids):
        output = self.model.generate(input_ids)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return {'answer': text}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)