#!/usr/bin/env bash
rm -rf nohup.out
#nice --19 nohup python3 train2.py &
date >start-time
nohup python3 train2.py && date >end-time &
