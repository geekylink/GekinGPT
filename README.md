# PicoGPT
A very basic set of scripts to train a GPT model on input.txt text and generate text based on a prompt for the trained model.

This should work on older hardware, such as 10XX series cards, and I think it should work on 9XX as well, not sure how far much older would work though, let me know if you try. 

# Train

First thing you need to do is train your model, this is two steps. The first is to tokenize the input data, then you need train for however many epochs

### Prepare - Tokenize your data


Run `train.py` with `--prepare` and `--input input.txt` to tokenize data and prepare for training epochs. 

Creates a directory: `out/output.model/`

```bash
python train.py --prepare --input input.txt [out/output.model]
```

Optional note: you can pass `--model` to use any other model provided by https://huggingface.co/models
You can use it like so: `--model gpt2-xl`

### Train - Run, run, run...

Train for X epochs using input.model and save to output.model Then train again for more epochs until coherent.
--model `out/output.model` and `out/output.model` should be the same model to resume and continue training. 

If you wish to save a newly trained model to a new `out/output2.model/` path you should copy the tokenized output from the first step into the new output directory

Note: change `--batch-size` for smaller/larger GPUs, default is 4.

```bash
python train.py --model [out/output.model] --epochs X [out/output.model]
```

# Generate text with the model

```bash
python run.py [out/output.model] <prompt_text>
```
