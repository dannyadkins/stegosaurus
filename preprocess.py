from transformers import GPT2Tokenizer
from datasets import load_dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def prepare_data():
    dataset = load_dataset("aeslc", "en")
    dataset = dataset["train"].map(tokenize)
    return dataset

def tokenize(item):
    text = item['email_body']
    tokenized = tokenizer._tokenize("<|txt|> " + text + " <|cph|> ")
    return {"tokens": tokenized}


if __name__ == "__main__":
    print(prepare_data())