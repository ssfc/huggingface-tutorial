from transformers import pipeline
# we recommend using the Auto* classes instead, as these are by design architecture-agnostic. While the previous code sample limits users to checkpoints loadable in the CamemBERT architecture, using the Auto* classes makes switching checkpoints simple
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import CamembertTokenizer, CamembertForMaskedLM


camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
print(results)

# tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# model = CamembertForMaskedLM.from_pretrained("camembert-base")

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")




