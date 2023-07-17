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


def train(inFile: str, pathSave: str):

    device = getCudaDevice()
    if device == None:
        return

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("Selecting cuda device...")
    model = model.to(device)  # This moves the model to GPU if available

    print("selecting optimizer...")
    optimizer = AdamW(model.parameters(), lr=1e-5)  # Define the optimizer, in this case, AdamW.

    print("\nLoading input data...")
    txtData = ""
    chunk_size = 4096  # Choose chunk size according to your memory constraints

    with open(inFile) as f:
        txtData = f.read()

    print("Input data loaded.")
    print("Input length:", len(txtData))
    print("Chunking data...")
    chunks = [txtData[i:i+chunk_size] for i in range(0, len(txtData), chunk_size)]

    print("Tokenizing data...")
    dataset = TextDataset(chunks, tokenizer)
    print("Data loader...")
    dataloader = DataLoader(dataset, batch_size=2)  # Adjust batch_size to fit your GPU

    num_epochs = 1

    for epoch in range(num_epochs):
        print("Working on epoch", epoch)
        for batch in dataloader:
            input_ids, attn_masks = batch
            input_ids = input_ids.to(device)
            attn_masks = attn_masks.to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)

            loss = outputs.loss
            print("Loss:", loss)
            loss.backward()
            optimizer.step()

    print("saving model...")
    model.save_pretrained(pathSave)

    print("Saving tokenizer...")
    tokenizer.save_pretrained(pathSave)


    # Move output to CPU for decoding
    #print("moving output to cpu...")
    #outputs = outputs.cpu()

    #print("generating outputs...")
    #for i in outputs:
    #        print(tokenizer.decode(i, skip_special_tokens=True))

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
    print("GekinGPT v0.0.1")

    args = sys.argv

    if (len(args) != 3):
        print("Usage:")
        print("\tpython train.py input.txt out/path.model")
        exit()

    inFile = args[1]
    pathSave = args[2]

    print("Training with input:", inFile)
    print("Let's gooo....\n")

    try:
        train(inFile, pathSave)
        #run(args[1], args[2])

    except FileNotFoundError:
        print("\nERROR: Input file not found:", inFile)
    finally:
        print("Exiting...")


    #run()
