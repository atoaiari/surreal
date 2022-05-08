#!/bin/bash

START_IDX=${1:-'0'}
END_IDX=${2:-'0'}
PYTHON_ARGS=${@:3}
JOB_PARAMS=${PYTHON_ARGS:-'--orientations 16 --frames 1 --textures 8 --background 10 --shapes 4'} 
BASIC_CONFIG_FILE=/home/atoaiari/work/surreal/datageneration/configs/basic_config_detangle_dataset

CONDA_PYTHON=/home/atoaiari/miniconda3/envs/detangle/bin/python
PYTHONPATH="" $CONDA_PYTHON generate_config.py $JOB_PARAMS $BASIC_CONFIG_FILE

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

# for i in {21..30}; do
#     $BLENDER_PATH/blender -b -t 12 --verbose 0 -P main_detangle.py -- --idx $i --orientations 16 --frames 2 
# done
    
for i in $(seq $START_IDX $END_IDX); do
    $BLENDER_PATH/blender -b --verbose 0 -P main_detangle.py -- --idx $i --gender female $BASIC_CONFIG_FILE
    $BLENDER_PATH/blender -b --verbose 0 -P main_detangle.py -- --idx $i --gender male $BASIC_CONFIG_FILE
done

# $BLENDER_PATH/blender -b --verbose 0 -P main_detangle.py -- --idx 0 $BASIC_CONFIG_FILE