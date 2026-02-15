#!/bin/bash

docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  tensorflow-keras-rnn

