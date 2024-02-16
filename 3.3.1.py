from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
import evaluate
import numpy as np


training_args = TrainingArguments("test-trainer")


checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


raw_datasets = load_dataset("glue", "mrpc")
print("raw_datasets", raw_datasets)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,  # 选了一个Auto二分model
    training_args,  # 名字是test-trainer
    train_dataset=tokenized_datasets["train"],  # glue中的train
    eval_dataset=tokenized_datasets["validation"],  # glue中的validation
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
# {'train_runtime': 37.6836, 'train_samples_per_second': 292.01, 'train_steps_per_second': 36.541, 'train_loss': 0.5535052374990072, 'epoch': 3.0}

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "mrpc")
print(metric.compute(predictions=preds, references=predictions.label_ids))

# 未训练的结果1：{'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
# 未训练的结果2：{'accuracy': 0.5563725490196079, 'f1': 0.6629422718808194}
# 训练后的结果1：{'accuracy': 0.8627450980392157, 'f1': 0.903448275862069}
# 训练后的结果2：{'accuracy': 0.8578431372549019, 'f1': 0.901023890784983}






