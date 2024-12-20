import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Define custom text inputs
custom_texts = ["NLP is based on transformers."]

# Tokenize custom text inputs
inputs = tokenizer(custom_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Generate text
generated_text = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=50,
    pad_token_id=tokenizer.pad_token_id,
    top_p=0.9,  # Nucleus sampling (probability threshold)
    temperature=0.7,  # Control randomness
    no_repeat_ngram_size=2  # Avoid repeating n-grams
)

# Decode and print generated text
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(f"Generated Text: {decoded_text}")
