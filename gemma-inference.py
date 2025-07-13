import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 tokenizer 和 model
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()

# 编写 prompt
# prompt = "Question: If Tom has 3 apples and gives 1 away, how many apples does he have?\nAnswer:"
prompt = "Introduce yourself:"

# 编码
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推理
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# 解码输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
