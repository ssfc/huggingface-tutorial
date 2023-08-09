import torch

# 3.3.1 Training
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification

training_args = TrainingArguments("test-trainer")


checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)






