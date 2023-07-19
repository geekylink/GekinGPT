import sys
import time
import torch
import datetime
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

VERSION="0.0.1"

START_DELAY=5 # Delay before spinning up your GPU so you can quick confirm settings, set to 0 to skip

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
    """
        Return CUDA device
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use 'cuda:0' if you want to specify GPU number

    if torch.cuda.is_available():
        print("Detected Cuda device")
    else:
        print("ERROR: No Cuda device detected.")
        return None

    return device

def loadModel(device = None, inModel: str = ""):
    """
        Load model path if provided, otherwise default to GPT2
    """
    modelPath = inModel if inModel != "" else "gpt2"

    print("Loading tokenizer:", modelPath ,"...")
    tokenizer = GPT2Tokenizer.from_pretrained(modelPath)
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading model", modelPath ,"...")
    model = GPT2LMHeadModel.from_pretrained(modelPath)

    # This moves the model to GPU if available
    if device:
        print("Selecting cuda device...")
        model = model.to(device)  

    return [model, tokenizer]


def loadData(inFile: str, chunkSize = 4096):
    """
        Loads data and chunks it
        Choose chunk size according to your memory constraints
    """
    print("\nLoading input data...")
    txtData = ""

    print("Loading file:", inFile)
    with open(inFile) as f:
        txtData = f.read()

    dataLen = len(txtData)
    dataKB = round(dataLen/1024, 2)
    dataMB = round(dataLen/1024/1024, 2)
    dataGB = round(dataLen/1024/1024/1024, 5)
    dataTB = round(dataLen/1024/1024/1024/1024, 6)

    print("Input data loaded.")
    print("Input length:", dataLen, "bytes")
    print(dataKB, "\tKB")
    print(dataMB, "\tMB")
    if dataMB > 1:
        print(dataGB, "\tGB")
    if dataGB > 0.1:
        print(dataTB, "\tTB")

    print("")
    print("Chunking data into chunks of size:", chunkSize)
    chunks = [txtData[i:i+chunkSize] for i in range(0, len(txtData), chunkSize)]

    return chunks

def tokenize(tokenizer, chunks, batchSize=2):
    """
        Tokenizes the data
        Adjust batchSize to fit your GPU
    """
    print("==============================")
    print("Tokenizing data... @", datetime.datetime.now())
    print("==============================")

    dataset = TextDataset(chunks, tokenizer)

    print("Data loader...")
    dataloader = DataLoader(dataset, batchSize)  

    print("==============================")
    print("Data loaded. @", datetime.datetime.now())
    print("==============================")

    return dataloader


def doEpochs(device, model, tokenizer, dataloader, numEpochs: int = 3):

    outEvery = 100 # Output status every 'outEvery' batch
    it = 0

    print("selecting optimizer...")
    optimizer = AdamW(model.parameters(), lr=1e-5)  # Define the optimizer, in this case, AdamW.

    for epoch in range(numEpochs):
        it = 0
        totalIt = len(dataloader)

        print("==============================")
        print("Starting Epoch", epoch+1, " of ", numEpochs ," @", datetime.datetime.now())
        print("==============================")

        print("Batches:", totalIt)

        for batch in dataloader:

            it += 1

            if it % outEvery == 0:
                print("Batch", it, "of", totalIt) 

            input_ids, attn_masks = batch
            input_ids = input_ids.to(device)
            attn_masks = attn_masks.to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

    print("=====================================================")
    print("Done with epochs, did ", numEpochs, " epochs. Completed @", datetime.datetime.now())
    print("=====================================================")

    return [model, tokenizer]

def saveModel(model = None, tokenizer = None, outModel: str  = "out/PicoGPT-unnamed.model"):

    if model:
        print("saving model...")
        model.save_pretrained(outModel)

    if tokenizer:
        print("Saving tokenizer...")
        tokenizer.save_pretrained(outModel)


def train(inFile: str, pathSave: str, inModel: str = "", numEpochs: int = 3, batchSize: int = 4):

    device = getCudaDevice()
    if device == None:
        return

    # Load the model
    model, tokenizer = loadModel(device, inModel)

    # Load and chunk the data to fit into memory
    chunks = loadData(inFile, 4096)

    dataloader = tokenize(tokenizer, chunks, batchSize=batchSize)

    # Train for epochs
    print("Starting training routine...")
    model, tokenizer = doEpochs(device, model, tokenizer, dataloader, numEpochs)

    saveModel(model, tokenizer, outModel)


    # Move output to CPU for decoding
    #print("moving output to cpu...")
    #outputs = outputs.cpu()

    #print("generating outputs...")
    #for i in outputs:
    #        print(tokenizer.decode(i, skip_special_tokens=True))

def parseArgs():
    """
        CLI Args for training and validation
    """
    parser = argparse.ArgumentParser(prog="PicoGPT.py", description="Train, finetune, and generate text with GPT models, even on older hardware like 10XX and earlier.", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--prepare', '-p', dest='prepare', action='store_const', 
                        const=True, default=False, 
                        help='''It is necessary to prepare the dataset for first time training, chunks and tokenizes.
On resuming training for additional epochs, prepare is not necessary.
''')

    parser.add_argument('--input', '-i', metavar='input.txt', dest='input', action='store',
                        default="",
                        help='Input text, necessary to provide for initial --prepare Optional on resume')

    parser.add_argument('--model', '-m', metavar='model', dest='inModel', action='store',
                        default="gpt2",
                        help='Input model, provide a path to a trained model to resume training. (Default: gpt2)')

    parser.add_argument('--epochs', '-e', metavar='epochs', dest='epochs', action='store',
                        default=3,
                        help='How many epochs to train for (Default: 3)')

    parser.add_argument('--batch-size', '-b', metavar='size', dest='batchSize', action='store',
                        default=4,
                        help='Batch size, adjust lower for GPUs with less memory (Default: 4)')

    parser.add_argument('outModel', help='''Necessary path to save trained model and tokenizer (Example: out/example.model)
This output path is also the same path you can use with --input/-i to resume training.
''')



    args = parser.parse_args()

    if args.prepare and not args.input and args.input != "":
        raise Exception("--prepare/-p must also have an --input/-i")

    return args

if __name__ == "__main__":

    args = parseArgs()

    isPrepare = args.prepare
    inModel   = args.inModel
    inFile    = args.input
    outModel  = args.outModel
    numEpochs = args.epochs
    batchSize = args.batchSize

    try:
        print("PicoGPT v" + str(VERSION))
        print("")

        print("Config for training:")
        print("====================")
        print("Importing model:", inModel)

        if args.input:
            print("Training with input data:", inFile)

        print("Saving output model to:", outModel)
        print("Training for", numEpochs,"epochs with a batch size of", batchSize)
        print("====================")

        print("")
        print("Starting soon. CTRL+C now if you changed your mind.")
        print("Starting in...", end='')
        for i in range(START_DELAY):
            print((START_DELAY-i), ".. ", end='', flush=True)
            time.sleep(1)

        print("Let's gooo....\n")
        #print(inFile, outModel, inModel, numEpochs, batchSize)
        train(inFile, outModel, inModel, numEpochs, batchSize) 

    except FileNotFoundError:
        print("\nERROR: Input file not found:", inFile)
    finally:
        print("Exiting...")

