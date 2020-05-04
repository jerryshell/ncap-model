#!/usr/bin/env bash
rm -rf nohup.out
date >start-time
nohup python3 train_function.py && date >end-time &
