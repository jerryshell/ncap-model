#!/usr/bin/env bash
rm -rf nohup.out
#nice --19 nohup python3 train_function.py &
date >start-time
nohup python3 retrain.py && date >end-time &
