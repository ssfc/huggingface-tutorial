from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "marian-finetuned-kde4-en-to-fr/checkpoint-11824"
translator = pipeline("translation", model=model_checkpoint)
print(translator("Default to expanded threads"))

print(translator("Unable to import %1 using the OFX importer plugin. This file is not the correct format."))

