from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "marian-finetuned-kde4-en-to-fr/checkpoint-11824"
translator = pipeline("translation", model=model_checkpoint)
print(translator("Default to expanded threads"))


