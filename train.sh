#!/usr/bin/env bash
rm -rf nohup.out
date >start-time
nohup python3 model_train.py 32 100 && date >end-time &
