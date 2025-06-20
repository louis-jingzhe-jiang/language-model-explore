import torch
from transformers import BertTokenizerFast
from datasets import load_dataset

if __name__ == "__main__":
    # initialize tokenizer and IMDB dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", 
        cache_dir="/Volumes/WD_SN750SE/Projects/language-model-explore/tokenizers")
    dataset = load_dataset("stanfordnlp/imdb", 
        cache_dir="/Volumes/WD_SN750SE/Projects/language-model-explore/datasets")
    
    # processing the datset
    tokens = tokenizer(dataset["train"]["text"], padding="max_length", truncation=True, 
                       max_length=256, return_tensors="pt")
    torch.save(tokens["input_ids"], "train_input.pt")
    torch.save(tokens["attention_mask"], "train_mask.pt")
    labels = torch.tensor(dataset["train"]["label"])
    result = torch.nn.functional.one_hot(labels, num_classes=2).float()
    torch.save(result, "train_label.pt")
    tokens = tokenizer(dataset["test"]["text"], padding="max_length", truncation=True, 
                       max_length=256, return_tensors="pt")
    torch.save(tokens["input_ids"], "test_input.pt")
    torch.save(tokens["attention_mask"], "test_mask.pt")
    labels = torch.tensor(dataset["test"]["label"])
    result = torch.nn.functional.one_hot(labels, num_classes=2).float()
    torch.save(result, "test_label.pt")