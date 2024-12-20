import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Load data
DATA_PATH = "train.csv"
data = pd.read_csv(DATA_PATH)
texts = data["full_text"].values[:500]  # Use first 500 samples

# Load model and tokenizer
MODEL_NAME_OR_PATH = "distilbert-base-uncased"
MAX_LENGTH = 512

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME_OR_PATH)

# Define Dataset class
class PretrainingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        tokenized = self.tokenizer(
            text=text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze(),
        }

# Split dataset
train_texts, eval_texts = train_test_split(texts, test_size=0.2, random_state=42)

# Create datasets
train_dataset = PretrainingDataset(train_texts, tokenizer, max_length=MAX_LENGTH)
eval_dataset = PretrainingDataset(eval_texts, tokenizer, max_length=MAX_LENGTH)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    report_to="none",  # Disable W&B
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
