from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


# load model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# load dataset
dataset = load_dataset('csv', data_files={'train': 'data-classification.csv'}, delimiter=',')
encoded_dataset = dataset.map(preprocess_function, batched=True)


# train model
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    # evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
)

trainer.train()

# 保存模型和Tokenizer
model.save_pretrained('./my_model')
tokenizer.save_pretrained('./my_model')

# 加载训练好的模型和tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./my_model')
tokenizer = AutoTokenizer.from_pretrained('./my_model')


# test model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 把模型放到GPU或CPU
model.eval()

sentence = "This movie is awesome!"

inputs = tokenizer(
    sentence,
    return_tensors="pt",
    truncation=True,
    padding='max_length',
    max_length=128
)
# 把输入也放到同一设备
for k in inputs:
    inputs[k] = inputs[k].to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=-1).item()

label_map = {0: "negative", 1: "positive"}
print("预测类别:", predicted_class_id)
print("情感为:", label_map[predicted_class_id])
