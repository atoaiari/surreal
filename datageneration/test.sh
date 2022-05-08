#!/bin/bash

START_IDX=${1:-'0'}
END_IDX=${2:-'0'}
PYTHON_ARGS=${@:3}
JOB_PARAMS=${PYTHON_ARGS:-'--orientations 16 --frames 2 --textures 8 --background 10 --shapes 5'} 

echo $START_IDX
echo $END_IDX
echo $JOB_PARAMS

for i in $(seq $START_IDX $END_IDX); do
    echo $i
done


