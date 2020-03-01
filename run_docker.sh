#!/usr/bin/env bash
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./script.py
