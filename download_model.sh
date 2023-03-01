#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "You must enter the model name as a parameter, e.g.: sh download_model.sh 117M"
    echo "Available models are: 117M 345M 774M 1558M"
    exit 1
fi

model=$1

mkdir -p models/$model

for filename in checkpoint encoder.json hparams.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta vocab.bpe; do
  fetch=$model/$filename
  echo "Fetching $fetch"
  curl --output models/$fetch https://openaipublic.blob.core.windows.net/gpt-2/models/$fetch
done
