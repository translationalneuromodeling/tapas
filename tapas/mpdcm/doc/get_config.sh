#! /bin/bash

# aponteeduardo@gmail.com
# Copyright (C)

set -e


gcc -v |& tail -n 1
cat /etc/issue.net
nvcc --version | tail -n 1
lshw -numeric -C display 2>&1 | grep product
matlab -nodisplay -r "fprintf(1, version); fprintf(1, '\n'); exit;" | tail -n 2| head -n 1
