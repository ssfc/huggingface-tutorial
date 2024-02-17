from transformers import AutoModelForSeq2SeqLM


model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)








