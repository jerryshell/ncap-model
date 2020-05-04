#!/usr/bin/env bash
rm -rf nohup.out
date >start-time
nohup python3 retrain.py && date >end-time &
