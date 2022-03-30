#!/bin/bash

JOB_PARAMS=${1:-'--idx 3'}

# SET PATHS HERE
FFMPEG_PATH=/usr/bin/ffmpeg
BLENDER_PATH=/home/atoaiari/work/surreal/blender/blender-2.78a-linux-glibc211-x86_64

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
SCIPY_PYTHON=/home/atoaiari/miniconda3/envs/surreal3.5/lib/python3.5/site-packages/
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.5:${BUNDLED_PYTHON}/lib/python3.5/site-packages:${SCIPY_PYTHON}:${PYTHONPATH}

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}

$BLENDER_PATH/blender -b -t 8 --verbose 0 -P main_detangle.py -- ${JOB_PARAMS}
