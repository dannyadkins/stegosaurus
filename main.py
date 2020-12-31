import preprocess
from transformers import GPT2Model

def main():
    dataset = preprocess.prepare_data()
    model = GPT2Model.from_pretrained('gpt2-large')
    print("Input tokens:\n", dataset[0]["tokens"])
    cipher = model.forward(dataset[0]["tokens"])
    print(cipher)

if __name__ == "__main__":
    main()