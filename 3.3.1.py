from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")


from transformers import AutoModelForSequenceClassification
checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

