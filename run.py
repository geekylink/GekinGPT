import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

class TextDataset(Dataset):
    def __init__(self, txt_list, tokenizer):
        self.input_ids = []
        self.attn_masks = []
        for txt in txt_list:
            inputs = tokenizer.encode_plus(txt, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            self.input_ids.append(inputs['input_ids'])
            self.attn_masks.append(inputs['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx].squeeze(), self.attn_masks[idx].squeeze()

def getCudaDevice():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use 'cuda:0' if you want to specify GPU number

    if torch.cuda.is_available():
        print("Detected Cuda device")
    else:
        print("ERROR: No Cuda device detected.")
        return None

    return device


def run(modelPath: str, prompt: str):
    #device = getCudaDevice()
    #if device == None:
    #    return

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(modelPath)
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(modelPath)
    print("Selecting cuda device...")

    # set the model in evaluation mode
    model.eval()

    # Encode the input text to tensor of integers by using the tokenizer
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text until the word count reaches max_length
    output = model.generate(inputs, max_length=500, do_sample=True, temperature=0.7)

    # Now decode the output tensor to readable string
    output_text = tokenizer.decode(output[0])

    print(output_text)


if __name__ == "__main__":
    print("PicoGPT v0.0.1")

    args = sys.argv

    if (len(args) != 3):
        print("Usage:")
        print("\tpython run.py [MODEL-PATH] <PROMPT>")
        exit()

    inFile = args[1]
    prompt = args[2]

    print("Generating with model:", inFile)
    print("Let's gooo....\n")

    try:
        run(inFile, prompt)

    except FileNotFoundError:
        print("\nERROR: Input file not found:", inFile)
    finally:
        print("Exiting...")


    #run()
