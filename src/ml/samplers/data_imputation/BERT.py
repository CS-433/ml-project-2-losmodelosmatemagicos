# TODO create of model based on BERT
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch

# Scaled numerical data (as previously scaled)
scaled_data = [1234, 567, 890, 2345, 123]

# Load BERT tokenizer and model
model_name = 'bert-base-uncased'  # or specify other variations
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
nlp_fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)

# Convert scaled numerical data to token IDs
token_ids = torch.tensor(scaled_data).unsqueeze(0)  # Assuming one batch of data

# Generate synthetic text using BERT's masked language modeling
outputs = model.generate(token_ids)
synthetic_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Synthetic Text:", synthetic_text)
