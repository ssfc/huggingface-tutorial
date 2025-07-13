import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型名称
model_name = "mistralai/Mistral-7B-v0.1"

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # 或 float16 取决于显卡
    device_map="auto"
)
model.eval()

# 编写 prompt
prompt = "What is the capital of France?"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推理
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

# 解码输出
print(tokenizer.decode(output[0], skip_special_tokens=True))
