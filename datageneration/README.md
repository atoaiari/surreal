# Create your own synthetic data #

## Preparation ##

<!-- Create the conda enviroment. -->
Create a new conda enviroment with python version 3.5: 
``` shell
$ conda create -n surreal python=3.5
```

<!-- Set the SMPL data. -->
You need to download SMPL for MAYA from http://smpl.is.tue.mpg.de in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following two files:
``` shell
basicModel_f_lbs_10_207_0_v1.0.2.fbx
basicModel_m_lbs_10_207_0_v1.0.2.fbx
```

Place these two files under `datageneration/smpl_data` folder.

With the same credentials as with the SURREAL dataset, you can download the remaining necessary SMPL data and place it in `datageneration/smpl_data`.
``` shell
$ ./download_smpl_data.sh /path/to/smpl_data yourusername yourpassword
```

<!-- Set Blender. -->
*Known problem: Blender2.78a has problems with pip. You can try with new versions of Blender. Otherwise, you can install the dependencies such as `scipy` to a new python3.5 environment and add this environment's `site-packages` to `PYTHONPATH` before running Blender.*

You need to download [Blender](http://download.blender.org/release/) and install scipy package to run the first part of the code. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2), which is shipped with its own python executable as well as distutils package. Therefore, it is sufficient to do the following:
``` shell
# Download Blender for Linux.
$ wget http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2
$ tar -xf blender-2.78a-linux-glibc211-x86_64.tar.bz2
$ pip install scipy==1.2.0
```

The file type for some of the temporary outputs from Blender will be EXR images. In order to read these images, the code uses OpenEXR bindings for Python, so:
<!-- Set OpenEXR. -->
``` shell
$ pip install openexr
```

## Running the code ##
Copy the `config.copy` into `config` and edit the `bg_path`, `tmp_path`, and `output_path`.

`run.sh` script is ran for each clip. You need to set FFMPEG_PATH, X264_PATH (optional), and BLENDER_PATH variables. `-t 1` option can be removed to run on multi cores, it runs faster.
