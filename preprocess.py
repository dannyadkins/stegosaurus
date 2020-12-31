from transformers import GPT2Tokenizer
from datasets import load_dataset


def prepare_data():
    dataset = load_dataset("wikicorpus")
    return dataset

if __name__ == "__main__":
    print(prepare_data)