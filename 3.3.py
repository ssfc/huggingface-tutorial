from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
# 3.3.1 Training
# 具体来说，`TrainingArguments` 类允许您设置以下类型的参数：
# - `output_dir`: 训练过程中保存模型和输出文件的目录。
# - `per_device_train_batch_size`: 每个设备的训练批量大小。
# - `num_train_epochs`: 训练的轮数。
# - `learning_rate`: 学习率。
# - `logging_dir`: 日志文件保存的目录。
# - `save_total_limit`: 保存的检查点总数限制。
from transformers import TrainingArguments

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments("test-trainer")


checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,  # 要训练的模型。
    training_args,  # 一个 TrainingArguments 类的实例，用于配置训练参数，如批量大小、学习率、训练轮数等。
    train_dataset=tokenized_datasets["train"],  # 训练数据集。
    eval_dataset=tokenized_datasets["validation"],  # 验证数据集。
    data_collator=data_collator,  # 数据收集器，用于将批次数据转换为模型输入。
    tokenizer=tokenizer,  # 用于将原始文本转换为模型输入的分词器。
)




