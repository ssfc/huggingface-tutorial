from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import torch

# 1. 加载 tokenizer 和模型（GPT2）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # gpt2 默认没有pad_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 构造小型CoT数据集
# 如果是 json 格式
dataset = load_dataset('json', data_files='cot_data.json', split='train')


# 3. 预处理函数
def preprocess(example):
    prompt = f"Question: {example['question']}\nAnswer: "
    target = example["cot"]
    input_text = prompt + target
    tokenized = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

# 4. 微调配置
training_args = TrainingArguments(
    output_dir="./cot-gpt2",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    logging_steps=1,
    save_steps=5,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# 5. 微调
trainer.train()


model.eval()
prompt = "Question: If a pen costs $2 and you buy 3 pens, how much do you spend?\nAnswer: "
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}  # ✅ 移动到同一设备

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
