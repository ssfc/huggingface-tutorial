import torch

# 3.3.1 Training
# 具体来说，`TrainingArguments` 类允许您设置以下类型的参数：
# - `output_dir`: 训练过程中保存模型和输出文件的目录。
# - `per_device_train_batch_size`: 每个设备的训练批量大小。
# - `num_train_epochs`: 训练的轮数。
# - `learning_rate`: 学习率。
# - `logging_dir`: 日志文件保存的目录。
# - `save_total_limit`: 保存的检查点总数限制。
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification

training_args = TrainingArguments("test-trainer")


checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)






