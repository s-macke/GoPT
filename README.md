# GoPT

This is my learning project to understand transformer models.
In particular, it implements GPT-2 model inference. It is the same model type
used for GPT-3, OpenAI Codex and ChatGPT, only smaller and therefore less powerful.

The model is implemented in pure Go, without any external dependencies and not optimized for speed. 

## Build

```bash
go build
```

## Download and convert small model

```bash
./download_model.sh 117M
python3 gpt2convert.py models/117M gpt2_117M.bin
./GoPT
```

## Download and convert large model

 ```bash
./download_model.sh 1558M
python3 gpt2convert.py model/1558M  gpt2_1558M.bin
./GoPT large
```

## Credits

This code is based on the following projects:

* https://github.com/viznut/vzgpt (the layer inference part)
* https://bellard.org/libnc/libnc.html (The transformation of python matrices to binary files. The original gpt2tc seems to be no longer available)
* https://github.com/ggerganov/ggml (just for reference)
