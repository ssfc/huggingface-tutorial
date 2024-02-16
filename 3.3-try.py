from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments


training_args = TrainingArguments("test-trainer")  # test-training是路径


checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


raw_datasets = load_dataset("glue", "sst2")
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



