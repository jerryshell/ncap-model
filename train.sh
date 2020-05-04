#!/usr/bin/env bash
rm -rf nohup.out
date >start-time
nohup python3 train.py && date >end-time &
