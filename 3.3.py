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
import numpy as np


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

trainer.train()

# 3.3.2 Evaluation
# predict(): 这是 Trainer 类中的一个方法，用于对给定的数据集进行预测。在这个例子中，使用 tokenized_datasets["validation"] 作为验证数据集。
# predictions: 这是一个包含预测结果的对象。它可能是一个包含预测标签、概率分数等的数据结构，具体取决于模型和任务。
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
# predictions.predictions: 这是一个包含模型对数据集样本的预测输出的数组。通常，这个数组的形状是 (样本数量, 类别数量)，其中每个元素表示模型对某个类别的预测分数。
# 使用 NumPy 库中的 argmax 函数，对预测输出进行求最大值索引的操作。axis=-1 表示在最后一个维度上执行操作，即在类别数量这个维度上进行求最大值索引。
# preds: 这是一个包含最可能的预测标签的数组。
# 例如，如果预测输出是 [[0.2, 0.6, 0.1], [0.8, 0.1, 0.1]]，那么 np.argmax(predictions.predictions, axis=-1) 的结果将是 [1, 0]，
# 表示模型预测的第一个样本最可能属于类别 1，第二个样本最可能属于类别 0。
preds = np.argmax(predictions.predictions, axis=-1)
print("preds:", preds)






