from transformers import pipeline


# 模型链接: https://huggingface.co/distilbert/distilgpt2
# generator = pipeline("text-generation", model="distilgpt2")
generator = pipeline("text-generation", model="gpt2")

result = generator(
    # "In this course, we will teach you how to",
    "I suck a girl's nipples",
    max_length=100,
    num_return_sequences=2,
)

print(result[0])
print(result[1])






