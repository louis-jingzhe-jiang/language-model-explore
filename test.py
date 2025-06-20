from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", 
    cache_dir="/Volumes/WD_SN750SE/Projects/language-model-explore/tokenizers")
text = "Hello, world! What is this?"
tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
# tokens includes: input_ids, attention_mask
print(len(tokens.input_ids[0]))