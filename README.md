# GoPT

GPT-2 Model inference

## Build

```bash
go build
```

## Download and convert small model

```bash
./download_model.sh 117M
python3 gpt2convert.py models/117M gpt2_117M.bin
```

## Download and convert large model

 ```bash
./download_model.sh 1558M
python3 gpt2convert.py model/1558M  gpt2_1558M.bin
 ```
You also have to change a one obvious line in main.go

