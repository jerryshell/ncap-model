#!/usr/bin/env bash
rm -rf nohup.out
date >start-time
nohup python3 model_retrain.py 'retrain.text_cnn_separable.2.h5' 32 10 && date >end-time &
