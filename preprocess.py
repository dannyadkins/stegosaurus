from transformers import GPT2Tokenizer
from datasets import load_dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def prepare_data():
    dataset = load_dataset("amazon_reviews_multi", "en")
    dataset = dataset["train"].map(tokenize)
    return dataset

def tokenize(item):
    text = item['review_body']
    return tokenizer.tokenize(text, padding=True, truncation=True)


if __name__ == "__main__":
    print(prepare_data())