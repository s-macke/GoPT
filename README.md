# GoPT

This is my learning project to understand transformer models.
In particular, it implements GPT-2 model inference. It is the same model type
used for ChatGPT, only smaller and therefore less powerful.

The model is implemented in pure Go, without any external dependencies and not optimized for speed. 

## Build

```bash
go build
```

## Download and convert small model
Download the model files 

* model.safetensors
* vocab.json

from huggingface at
https://huggingface.co/openai-community/gpt2/tree/main

```bash
wget -O model.safetensors "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors?download=true"
wget -O vocab.json "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json?download=true"
./GoPT
```

## Credits

This code is based on the following projects:

* https://github.com/viznut/vzgpt (the layer inference part)
* https://bellard.org/libnc/libnc.html (The transformation of python matrices to binary files. The original gpt2tc seems to be no longer available)
* https://github.com/ggerganov/ggml (just for reference)
