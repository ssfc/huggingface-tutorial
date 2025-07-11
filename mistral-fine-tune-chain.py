from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

# 1. ★更换为 Mistral-7B
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 指定pad_token
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 2. 读取数据集
dataset = load_dataset('json', data_files='cot_data.json', split='train')

# 3. 预处理函数
def preprocess(example):
    prompt = f"Question: {example['question']}\nAnswer: "
    target = example["cot"]
    input_text = prompt + target
    # ★建议设置 max_length 至少为 512
    tokenized = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

# 4. 微调配置（output_dir建议更名避免混淆）
training_args = TrainingArguments(
    output_dir="./cot-mistral7b",
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

# 6. 推理示例 (不用动)
model.eval()
prompt = "Question: If a pen costs \$2 and you buy 3 pens, how much do you spend?\nAnswer: "
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
