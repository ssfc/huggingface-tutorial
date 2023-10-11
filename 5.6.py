
# 5.6.1 Using embeddings for semantic search
from datasets import load_dataset

issues_dataset = load_dataset("lewtun/github-issues", split="train")
# 不知道为啥这里运行不起来
print("issues_dataset:", issues_dataset)