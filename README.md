# PicoGPT
A very basic set of scripts to train a GPT model on input.txt text and generate text based on a prompt for the trained model.

This should work on older hardware, such as 10XX series cards, and I think it should work on 9XX as well, not sure how far much older would work though, let me know if you try. 

# Train

First thing you need to do is train your model, you have a couple options for training:

```bash
# Train output.model using input.txt and GPT2
# Do this first to prepare your model and try quick trial, then continue training below. 
python train.py --input input.txt [out/output.model]

# Train for X epochs using input.model and input.txt and save to output.model
# Then train again for however many epochs until coherent
# Note: for input.model you should be able to select any provided model as well, such as gpt2
# Note: change batchSize for smaller/larger GPUs
python train.py --model [out/input.model] --input input.txt --epochs X --batchSize 4 [out/output.model]
```

# Generate text with the model

```bash
python run.py [out/output.model] <prompt_text>
```
