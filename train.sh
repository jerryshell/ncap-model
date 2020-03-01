#!/usr/bin/env bash
rm -rf nohup.out
nice --19 nohup python3 train2.py &
