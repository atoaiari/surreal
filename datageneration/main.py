import sys
import os
import random
import math
import bpy
import numpy as np
from os import getenv
from os import remove
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from glob import glob
from random import choice
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam
import scipy.io

# to read exr imgs
import OpenEXR 
import array
import Imath

import time

from utils.utils import *


sys.path.insert(0, ".")


def main():
    # time logging
    global start_time
    start_time = time.time()

    import argparse
    
    # parse commandline arguments
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    
    idx = args.idx
    ishape = args.ishape
    stride = args.stride
    
    log_message("input idx: %d" % idx)
    log_message("input ishape: %d" % ishape)
    log_message("input stride: %d" % stride)
    
    if idx == None:
        exit(1)
    if ishape == None:
        exit(1)
    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50
    
    # import idx info (name, split)
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    (runpass, idx) = divmod(idx, len(idx_info))
    
    log_message("runpass: %d" % runpass)
    log_message("output idx: %d" % idx)
    idx_info = idx_info[idx]
    log_message("sequence: %s" % idx_info['name'])
    log_message("nb_frames: %f" % idx_info['nb_frames'])
    log_message("use_split: %s" % idx_info['use_split'])

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    bg_path = params['bg_path']
    resy = params['resy']
    resx = params['resx']
    clothing_option = params['clothing_option'] # grey, nongrey or all
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']
    # openexr_py2_path = params['openexr_py2_path']

    # compute number of cuts
    nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride))/stride)))
    log_message("Max ishape: %d" % (nb_ishape - 1))
    
    if ishape == None:
        exit(1)
    
    assert(ishape < nb_ishape)
    
    # name is set given idx
    name = idx_info['name']
    output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    params['output_path'] = output_path
    tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))
    params['tmp_path'] = tmp_path
    
    # check if already computed
    #  + clean up existing tmp folders if any
    if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
        os.system('rm -rf %s' % tmp_path)
    rgb_vid_filename = "%s_c%04d.mp4" % (join(output_path, name.replace(' ', '')), (ishape + 1))
    #if os.path.isfile(rgb_vid_filename):
    #    log_message("ALREADY COMPUTED - existing: %s" % rgb_vid_filename)
    #    return 0
    
    # create tmp directory
    if not exists(tmp_path):
        mkdir_safe(tmp_path)
    
    # >> don't use random generator before this point <<

    # initialize RNG with seeds from sequence id
    import hashlib
    s = "synth_data:%d:%d:%d" % (idx, runpass,ishape)
    seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)
    
    if(output_types['vblur']):
        vblur_factor = np.random.normal(0.5, 0.5)
        params['vblur_factor'] = vblur_factor
    
    log_message("Setup Blender")

    # create copy-spher.harm. directory if not exists
    sh_dir = join(tmp_path, 'spher_harm')
    if not exists(sh_dir):
        mkdir_safe(sh_dir)
    sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (runpass, idx))
    os.system('cp spher_harm/sh.osl %s' % sh_dst)

    genders = {0: 'female', 1: 'male'}
    # pick random gender
    gender = choice(genders)

    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    log_message("Listing background images")
    bg_names = join(bg_path, '%s_img.txt' % idx_info['use_split'])
    nh_txt_paths = []
    with open(bg_names) as f:
        for line in f:
            nh_txt_paths.append(join(bg_path, line))

    # grab clothing names
    log_message("clothing: %s" % clothing_option)
    with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % ( gender, idx_info['use_split'] ) ) ) as f:
        txt_paths = f.read().splitlines()

    # if using only one source of clothing
    if clothing_option == 'nongrey':
        txt_paths = [k for k in txt_paths if 'nongrey' in k]
    elif clothing_option == 'grey':
        txt_paths = [k for k in txt_paths if 'nongrey' not in k]
    
    # random clothing texture
    cloth_img_name = choice(txt_paths)
    cloth_img_name = join(smpl_data_folder, cloth_img_name)
    cloth_img = bpy.data.images.load(cloth_img_name)

    # random background
    bg_img_name = choice(nh_txt_paths)[:-1]
    bg_img = bpy.data.images.load(bg_img_name)

    log_message("Loading parts segmentation")
    beta_stds = np.load(join(smpl_data_folder, ('%s_beta_stds.npy' % gender)))
    
    log_message("Building materials tree")
    mat_tree = bpy.data.materials['Material'].node_tree
    create_sh_material(mat_tree, sh_dst, cloth_img)
    res_paths = create_composite_nodes(scene.node_tree, params, img=bg_img, idx=idx)

    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
    
    log_message("Initializing scene")
    camera_distance = np.random.normal(8.0, 1)
    params['camera_distance'] = camera_distance
    ob, obname, arm_ob, cam_ob = init_scene(scene, params, gender)

    setState0()
    ob.select = True
    bpy.context.scene.objects.active = ob
    segmented_materials = True #True: 0-24, False: expected to have 0-1 bg/fg
    
    log_message("Creating materials segmentation")
    # create material segmentation
    if segmented_materials:
        materials = create_segmentation(ob, params)
        prob_dressed = {'leftLeg':.5, 'leftArm':.9, 'leftHandIndex1':.01,
                        'rightShoulder':.8, 'rightHand':.01, 'neck':.01,
                        'rightToeBase':.9, 'leftShoulder':.8, 'leftToeBase':.9,
                        'rightForeArm':.5, 'leftHand':.01, 'spine':.9,
                        'leftFoot':.9, 'leftUpLeg':.9, 'rightUpLeg':.9,
                        'rightFoot':.9, 'head':.01, 'leftForeArm':.5,
                        'rightArm':.5, 'spine1':.9, 'hips':.9,
                        'rightHandIndex1':.01, 'spine2':.9, 'rightLeg':.5}
    else:
        materials = {'FullBody': bpy.data.materials['Material']}
        prob_dressed = {'FullBody': .6}

    orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()) - Vector((-1., 1., 1.))
    orig_cam_loc = cam_ob.location.copy()

    # unblocking both the pose and the blendshape limits
    for k in ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    log_message("Loading body data")
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gender)
    
    log_message("Loaded body data for %s" % name)
    
    nb_fshapes = len(fshapes)
    if idx_info['use_split'] == 'train':
        fshapes = fshapes[:int(nb_fshapes*0.8)]
    elif idx_info['use_split'] == 'test':
        fshapes = fshapes[int(nb_fshapes*0.8):]
    
    # pick random real body shape
    shape = choice(fshapes) #+random_shape(.5) can add noise
    #shape = random_shape(3.) # random body shape
    
    # example shapes
    #shape = np.zeros(10) #average
    #shape = np.array([ 2.25176191, -3.7883464 ,  0.46747496,  3.89178988,  2.20098416,  0.26102114, -3.07428093,  0.55708514, -3.94442258, -2.88552087]) #fat
    #shape = np.array([-2.26781107,  0.88158132, -0.93788176, -0.23480508,  1.17088298,  1.55550789,  0.44383225,  0.37688275, -0.27983086,  1.77102953]) #thin
    #shape = np.array([ 0.00404852,  0.8084637 ,  0.32332591, -1.33163664,  1.05008727,  1.60955275,  0.22372946, -0.10738459,  0.89456312, -1.22231216]) #short
    #shape = np.array([ 3.63453289,  1.20836171,  3.15674431, -0.78646793, -1.93847355, -0.32129994, -0.97771656,  0.94531640,  0.52825811, -0.99324327]) #tall

    ndofs = 10

    scene.objects.active = arm_ob
    orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()

    # create output directory
    if not exists(output_path):
        mkdir_safe(output_path)

    # spherical harmonics material needs a script to be loaded and compiled
    scs = []
    for mname, material in materials.items():
        scs.append(material.node_tree.nodes['Script'])
        scs[-1].filepath = sh_dst
        scs[-1].update()

    rgb_dirname = name.replace(" ", "") + '_c%04d.mp4' % (ishape + 1)
    rgb_path = join(tmp_path, rgb_dirname)

    data = cmu_parms[name]
    
    fbegin = ishape*stepsize*stride
    fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
    
    log_message("Computing how many frames to allocate")
    N = len(data['poses'][fbegin:fend:stepsize])
    log_message("Allocating %d frames in mat file" % N)

    # force recomputation of joint angles unless shape is all zeros
    curr_shape = np.zeros_like(shape)
    nframes = len(data['poses'][::stepsize])

    matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
    log_message('Working on %s' % matfile_info)

    # allocate
    dict_info = {}
    dict_info['bg'] = np.zeros((N,), dtype=np.object) # background image path
    dict_info['camLoc'] = np.empty(3) # (1, 3)
    dict_info['clipNo'] = ishape +1
    dict_info['cloth'] = np.zeros((N,), dtype=np.object) # clothing texture image path
    dict_info['gender'] = np.empty(N, dtype='uint8') # 0 for male, 1 for female
    dict_info['joints2D'] = np.empty((2, 24, N), dtype='float32') # 2D joint positions in pixel space
    dict_info['joints3D'] = np.empty((3, 24, N), dtype='float32') # 3D joint positions in world coordinates
    dict_info['light'] = np.empty((9, N), dtype='float32')
    dict_info['pose'] = np.empty((data['poses'][0].size, N), dtype='float32') # joint angles from SMPL (CMU)
    dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (ishape + 1)
    dict_info['shape'] = np.empty((ndofs, N), dtype='float32')
    dict_info['zrot'] = np.empty(N, dtype='float32')
    dict_info['camDist'] = camera_distance
    dict_info['stride'] = stride

    if name.replace(" ", "").startswith('h36m'):
        dict_info['source'] = 'h36m'
    else:
        dict_info['source'] = 'cmu'

    if(output_types['vblur']):
        dict_info['vblur_factor'] = np.empty(N, dtype='float32')

    # for each clipsize'th frame in the sequence
    get_real_frame = lambda ifr: ifr
    random_zrot = 0
    reset_loc = False
    batch_it = 0
    curr_shape = reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                                       cam_ob, smpl_data['regression_verts'], smpl_data['joint_regressor'])
    # random_zrot = 2*np.pi*np.random.rand()
    
    arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()

    # create a keyframe animation with pose, translation, blendshapes and camera motion
    # LOOP TO CREATE 3D ANIMATION
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        iframe = seq_frame
        scene.frame_set(get_real_frame(seq_frame))

        # apply the translation, pose and shape to the character
        apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame))
        dict_info['shape'][:, iframe] = shape[:ndofs]
        dict_info['pose'][:, iframe] = pose

        # log_message("POSE")
        # log_message(dict_info['pose'][0:3] * 180/np.pi)
        # log_message(np.linalg.norm(dict_info['pose'][0:3])*180/np.pi)
        # log_message(dict_info['pose'][0:3] / np.linalg.norm(dict_info['pose'][0:3]))
        
        # log_message("PELVIS")
        # log_message(dict_info['pose'][3:6] * 180/np.pi)
        # log_message(np.linalg.norm(dict_info['pose'][3:6])*180/np.pi)

        
        dict_info['gender'][iframe] = list(genders)[list(genders.values()).index(gender)]
        if(output_types['vblur']):
            dict_info['vblur_factor'][iframe] = vblur_factor

        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
        dict_info['zrot'][iframe] = random_zrot

        scene.update()

        # Bodies centered only in each minibatch of clipsize frames
        if seq_frame == 0 or reset_loc: 
            reset_loc = False
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            cam_ob.location = orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())
            cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))
            dict_info['camLoc'] = np.array(cam_ob.location)

    scene.node_tree.nodes['Image'].image = bg_img

    for part, material in materials.items():
        material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

    # random light
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    sh_coeffs[0] = .5 + .9 * np.random.rand() # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[1] = -.7 * np.random.rand()

    for ish, coeff in enumerate(sh_coeffs):
        for sc in scs:
            sc.inputs[ish+1].default_value = coeff

    # iterate over the keyframes and render
    # LOOP TO RENDER
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        scene.frame_set(get_real_frame(seq_frame))
        iframe = seq_frame

        dict_info['bg'][iframe] = bg_img_name
        dict_info['cloth'][iframe] = cloth_img_name
        dict_info['light'][:, iframe] = sh_coeffs

        scene.render.use_antialiasing = False
        scene.render.filepath = join(rgb_path, 'Image%04d.png' % get_real_frame(seq_frame))

        log_message("Rendering frame %d" % seq_frame)
        
        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

        # NOTE:
        # ideally, pixels should be readable from a viewer node, but I get only zeros
        # --> https://ammous88.wordpress.com/2015/01/16/blender-access-render-results-pixels-directly-from-python-2/
        # len(np.asarray(bpy.data.images['Render Result'].pixels) is 0
        # Therefore we write them to temporary files and read with OpenEXR library (available for python2) in main_part2.py
        # Alternatively, if you don't want to use OpenEXR library, the following commented code does loading with Blender functions, but it can cause memory leak.
        # If you want to use it, copy necessary lines from main_part2.py such as definitions of dict_normal, matfile_normal...

        #for k, folder in res_paths.items():
        #   if not k== 'vblur' and not k=='fg':
        #       path = join(folder, 'Image%04d.exr' % get_real_frame(seq_frame))
        #       render_img = bpy.data.images.load(path)
        #       # render_img.pixels size is width * height * 4 (rgba)
        #       arr = np.array(render_img.pixels[:]).reshape(resx, resy, 4)[::-1,:, :] # images are vertically flipped 
        #       if k == 'normal':# 3 channels, original order
        #           mat = arr[:,:, :3]
        #           dict_normal['normal_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'gtflow':
        #           mat = arr[:,:, 1:3]
        #           dict_gtflow['gtflow_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'depth':
        #           mat = arr[:,:, 0]
        #           dict_depth['depth_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'segm':
        #           mat = arr[:,:,0]
        #           dict_segm['segm_%d' % (iframe + 1)] = mat.astype(np.uint8, copy=False)
        #
        #       # remove the image to release memory, object handles, etc.
        #       render_img.user_clear()
        #       bpy.data.images.remove(render_img)

        # bone locations should be saved after rendering so that the bones are updated
        bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
        dict_info['joints2D'][:, :, iframe] = np.transpose(bone_locs_2D)
        dict_info['joints3D'][:, :, iframe] = np.transpose(bone_locs_3D)

        reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

    # save a .blend file for debugging:
    # bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path, 'pre.blend'))
    
    # save RGB data with ffmpeg (if you don't have h264 codec, you can replace with another one and control the quality with something like -q:v 3)
    cmd_ffmpeg = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 ''%s_c%04d.mp4''' % (join(rgb_path, 'Image%04d.png'), join(output_path, name.replace(' ', '')), (ishape + 1))
    log_message("Generating RGB video (%s)" % cmd_ffmpeg)
    os.system(cmd_ffmpeg)
    
    if(output_types['vblur']):
        cmd_ffmpeg_vblur = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ''%s_c%04d.mp4''' % (join(res_paths['vblur'], 'Image%04d.png'), join(output_path, name.replace(' ', '')+'_vblur'), (ishape + 1))
        log_message("Generating vblur video (%s)" % cmd_ffmpeg_vblur)
        os.system(cmd_ffmpeg_vblur)
   
    if(output_types['fg']):
        cmd_ffmpeg_fg = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 ''%s_c%04d.mp4''' % (join(res_paths['fg'], 'Image%04d.png'), join(output_path, name.replace(' ', '')+'_fg'), (ishape + 1))
        log_message("Generating fg video (%s)" % cmd_ffmpeg_fg)
        os.system(cmd_ffmpeg_fg)
   
    # cmd_tar = 'tar -czvf %s/%s.tar.gz -C %s %s' % (output_path, rgb_dirname, tmp_path, rgb_dirname)
    # log_message("Tarballing the images (%s)" % cmd_tar)
    # os.system(cmd_tar)
    
    # save annotation excluding png/exr data to _info.mat file
    scipy.io.savemat(matfile_info, dict_info, do_compression=True)


    #### PART 2 ####
    log_message("Start part 2")
    res_paths = {k:join(tmp_path, '%05d_%s'%(idx, k)) for k in output_types if output_types[k]}
    
    # .mat files
    matfile_normal = join(output_path, name.replace(" ", "") + "_c%04d_normal.mat" % (ishape + 1))
    matfile_gtflow = join(output_path, name.replace(" ", "") + "_c%04d_gtflow.mat" % (ishape + 1))
    matfile_depth = join(output_path, name.replace(" ", "") + "_c%04d_depth.mat" % (ishape + 1))
    matfile_segm = join(output_path, name.replace(" ", "") + "_c%04d_segm.mat" % (ishape + 1))
    dict_normal = {}
    dict_gtflow = {}
    dict_depth = {}
    dict_segm = {}
    get_real_frame = lambda ifr: ifr
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # overlap determined by stride (# subsampled frames to skip)
    fbegin = ishape*stepsize*stride
    fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
    # LOOP OVER FRAMES
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        iframe = seq_frame
        
        log_message("Processing frame %d" % iframe)
        
        for k, folder in res_paths.items():
            if not k== 'vblur' and not k=='fg':
                path = join(folder, 'Image%04d.exr' % get_real_frame(seq_frame))
                exr_file = OpenEXR.InputFile(path)
                if k == 'normal':
                    mat = np.transpose(np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")], (3, resx, resy)), (1, 2, 0))
                    dict_normal['normal_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False) # +1 for the 1-indexing
                elif k == 'gtflow':
                    mat = np.transpose(np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G")], (2, resx, resy)), (1, 2, 0))
                    dict_gtflow['gtflow_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
                elif k == 'depth':
                    mat = np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R")], (resx, resy))
                    dict_depth['depth_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
                elif k == 'segm':
                    mat = np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R")], (resx, resy))
                    dict_segm['segm_%d' % (iframe + 1)] = mat.astype(np.uint8, copy=False)

    scipy.io.savemat(matfile_normal, dict_normal, do_compression=True)
    scipy.io.savemat(matfile_gtflow, dict_gtflow, do_compression=True)
    scipy.io.savemat(matfile_depth, dict_depth, do_compression=True)
    scipy.io.savemat(matfile_segm, dict_segm, do_compression=True)

    # cleaning up tmp
    # if tmp_path != "" and tmp_path != "/":
    #     log_message("Cleaning up tmp")
    #     os.system('rm -rf %s' % tmp_path)

    log_message("Completed batch")


if __name__ == '__main__':
    main()
