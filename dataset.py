from transformers import AutoTokenizer
from torch.utils.data import Dataset



class QADataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=512)
        self.data = data

    def __len__(self):
        return sum(len(item["QA"]) for item in self.data)

    def __getitem__(self, idx):
        item_idx = idx // len(self.data)
        qa_idx = idx % len(self.data[item_idx]["QA"])
        context = self.data[item_idx]["Context"]
        qa_pair = self.data[item_idx]["QA"][qa_idx]
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
            "answer": answer,
            "text": input_text
        }