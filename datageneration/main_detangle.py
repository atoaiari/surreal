from email.policy import default
import string
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
import bmesh

# to read exr imgs
import OpenEXR 
import array
import Imath

import time
import json
import argparse
import config

from utils.utils import *

sys.path.insert(0, ".")

def main():
    # parse commandline arguments
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images for disentanglement.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--frames', type=int, help='frames to use from the sequence', default=5)
    parser.add_argument('--gender', type=int,
                        help='-1: both, 0: female, 1: male', default=-1)
    parser.add_argument('--backgrounds', type=int,
                        help='number of backgrounds', default=10)
    parser.add_argument('--orientations', type=int, choices=[4, 8, 16], default=4,
                        help='number of orientation classes')
    parser.add_argument('--shapes', type=int, default=5,
                        help='number of shapes')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    
    idx = args.idx
    n_frames = args.frames
    n_backgrounds = args.backgrounds
    n_orientations = args.orientations
    n_shapes = args.shapes
    
    log_message("input idx: %d" % idx)
    log_message("frames used from the sequence: %d" %n_frames)
    log_message("number of backgrounds: %d" % n_backgrounds)
    log_message("number of orientations: %d" % n_orientations)
    log_message("number of shapes: %d" % n_shapes)
    
    if idx == None:
        exit(1)
    
    # import idx info (name, split)
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    (runpass, idx) = divmod(idx, len(idx_info))
    
    log_message("runpass: %d" % runpass)
    log_message("output idx: %d" % idx)
    idx_info = idx_info[idx]
    log_message("sequence: %s" % idx_info['name'])
    log_message("use_split: %s" % idx_info['use_split'])

    params = config.load_file('config_detangle', 'SYNTH_DATA')
    
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    bg_path = params['bg_path']
    resy = params['resy']
    resx = params['resx']
    clothing_option = params['clothing_option'] # grey, nongrey or all
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    stepsize = params['stepsize']
    
    tot_frames = int(idx_info['nb_frames'] * stepsize)
    assert(n_frames < tot_frames)
    
    # name is set given idx
    name = idx_info['name']
    info_files_path = join(output_path, "info")
    params['info_files_path'] = info_files_path
    output_path = join(output_path, '%s' % name.replace(" ", ""))
    params['output_path'] = output_path
    tmp_path = join(tmp_path, '%s' % (name.replace(" ", "")))
    params['tmp_path'] = tmp_path

    # check if already computed
    #  + clean up existing tmp folders if any
    if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
        os.system('rm -rf %s' % tmp_path)

    if exists(output_path) and output_path != "" and output_path != "/":
        os.system('rm -rf %s' % output_path)
    
    if not exists(info_files_path):
        mkdir_safe(info_files_path)

    # create tmp directory
    if not exists(tmp_path):
        mkdir_safe(tmp_path)

    # create output directory
    if not exists(output_path):
        mkdir_safe(output_path)

    jsonfile_info = join(info_files_path, "%04d-"%idx + name.replace(" ", "") + "-info.json")
    log_message('Working on %s' % jsonfile_info)

    # >> don't use random generator before this point <<

    # initialize RNG with seeds from sequence id
    # import hashlib
    # s = "synth_data:%d:%d" % (idx, runpass)
    # seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    #log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    seed_number = 11
    random.seed(seed_number)
    np.random.seed(seed_number)
    
    log_message("Setup Blender")

    # create copy-spher.harm. directory if not exists
    sh_dir = join(tmp_path, 'spher_harm')
    if not exists(sh_dir):
        mkdir_safe(sh_dir)
    sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (runpass, idx))
    os.system('cp spher_harm/sh.osl %s' % sh_dst)

    genders = {0: 'female', 1: 'male'}
    # set gender.
    if args.gender == -1:
        gender = [genders.get(g) for g in genders]
    else:
        gender = genders.get(args.gender)
    log_message("gender: %s" % gender)

    # set background
    bg_names = join(bg_path, '%s_img.txt' % idx_info['use_split'])
    nh_txt_paths = []
    with open(bg_names) as f:
        for line in f:
            nh_txt_paths.append(join(bg_path, line[:-1]))
    backgrounds = np.random.choice(nh_txt_paths[:-1], n_backgrounds, replace=False)

    # set orientations
    orientations = list(range(0, 360, int(360/n_orientations)))

    # random light
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    sh_coeffs[0] = .5 + .9 * np.random.rand() # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[1] = -.7 * np.random.rand()

    # select n_frames-1 random frames (the first is always 0)
    sequence_info = []
    frames = list(np.random.choice(np.arange(1, tot_frames), size=n_frames-1, replace=False))
    frames.append(0)
    frames.sort()

    # fixed camera distance
    # camera_distance = np.random.normal(8.0, 1)
    camera_distance = 7.2
    print("camera Distance: %f"%camera_distance)

    # fixed clothing texture
    cloth_img_name = ["textures/female/nongrey_female_0902.jpg", "textures/male/nongrey_male_0388.jpg"]

    img_ct = 0
    for igndr, gndr in enumerate(gender):
        # # grab clothing names
        # log_message("clothing: %s" % clothing_option)
        # # random clothing texture
        # with open(join(smpl_data_folder, 'textures', '%s_%s.txt' % (gndr, idx_info['use_split']))) as f:
        #     txt_paths = f.read().splitlines()
        # # if using only one source of clothing
        # if clothing_option == 'nongrey':
        #     txt_paths = [k for k in txt_paths if 'nongrey' in k]
        # elif clothing_option == 'grey':
        #     txt_paths = [k for k in txt_paths if 'nongrey' not in k]
        # cloth_img_name = choice(txt_paths)
        # cloth_img_name = join(smpl_data_folder, cloth_img_name)

        # fixed clothing texture
        cloth_img_name = join(smpl_data_folder, cloth_img_name[igndr])

        ### BLENDER ###
        scene = bpy.data.scenes['Scene']
        scene.render.engine = 'CYCLES'

        bpy.data.materials['Material'].use_nodes = True
        scene.cycles.shading_system = True
        scene.use_nodes = True
        
        cloth_img = bpy.data.images.load(cloth_img_name)
        log_message("clothing texture: %s" % cloth_img_name)
        
        #log_message("Building materials tree")
        mat_tree = bpy.data.materials['Material'].node_tree
        create_sh_material(mat_tree, sh_dst, cloth_img)

        #log_message("Loading smpl data")
        smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
        
        #log_message("Initializing scene")
        params['camera_distance'] = camera_distance

        ob, obname, arm_ob, cam_ob = init_scene(scene, params, gndr)

        setState0()
        ob.select = True
        bpy.context.scene.objects.active = ob
        segmented_materials = True #True: 0-24, False: expected to have 0-1 bg/fg
        
        #log_message("Creating materials segmentation")
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
        
        scene.objects.active = arm_ob
        orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()
        
        # spherical harmonics material needs a script to be loaded and compiled
        scs = []
        for mname, material in materials.items():
            scs.append(material.node_tree.nodes['Script'])
            scs[-1].filepath = sh_dst
            scs[-1].update()

        # log_message("Loading body data")
        cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gndr)   
        log_message("Loaded body data for %s" % name)
        
        nb_fshapes = len(fshapes)
        if idx_info['use_split'] == 'train':
            fshapes = fshapes[:int(nb_fshapes*0.8)]
        elif idx_info['use_split'] == 'test':
            fshapes = fshapes[int(nb_fshapes*0.8):] 
        # pick random real body shape
        # shape = choice(fshapes)     #+random_shape(.5) can add noise
        shapes_idx = np.random.choice(np.arange(len(fshapes)), size=n_shapes, replace=False)
        shapes = fshapes[shapes_idx]
        ndofs = 10
        
        data = cmu_parms[name]

        # for each clipsize'th frame in the sequence
        get_real_frame = lambda ifr: ifr
        random_zrot = 0
        # additional z rotation, not used
        # random_zrot = 2*np.pi*np.random.rand()
        reset_loc = False

        for ishape, shape in enumerate(shapes):
            curr_shape = reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                                            cam_ob, smpl_data['regression_verts'], smpl_data['joint_regressor'])
            
            arm_ob.animation_data_clear()
            cam_ob.animation_data_clear()

            for part, material in materials.items():
                material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

            for ish, coeff in enumerate(sh_coeffs):
                for sc in scs:
                    sc.inputs[ish+1].default_value = coeff

            # create a keyframe animation with pose, translation, blendshapes and camera motion
            # LOOP TO CREATE 3D ANIMATION
            for orientation in orientations:
                for background in backgrounds:        
                    bg_img = bpy.data.images.load(background)
                    res_paths = create_composite_nodes(scene.node_tree, params, img=bg_img, idx=idx)
                    
                    # i use the same translation for all the frames
                    translations = np.tile(data['trans'][0], (n_frames, 1))
                    for iframe, (seq_frame, pose, trans) in enumerate(zip(frames, data['poses'][frames], translations)):
                        scene.frame_set(get_real_frame(seq_frame))
                
                        # apply the translation, pose and shape to the character
                        apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame), orientation)
                        
                        dict_info = {}
                        dict_info['img_idx'] = img_ct
                        dict_info['frame'] = int(get_real_frame(seq_frame))
                        dict_info['index'] = idx
                        dict_info['sequence'] = name.replace(" ", "")
                        dict_info['camDist'] = camera_distance
                        if name.replace(" ", "").startswith('h36m'):
                            dict_info['source'] = 'h36m'
                        else:
                            dict_info['source'] = 'cmu'

                        dict_info['shape'] = shape[:ndofs]
                        dict_info['shape_idx'] = ishape
                        dict_info['pose'] = pose
                        dict_info['gender'] = gndr
                        dict_info['zrot'] = random_zrot
                        dict_info['orientation'] = orientation
            
                        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
                        arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
                        scene.update()

                        # Bodies centered only in each minibatch of clipsize frames
                        if seq_frame == 0 or reset_loc: 
                            reset_loc = False
                            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
                            cam_ob.location = orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())
                            cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))
                            dict_info['camLoc'] = np.array(cam_ob.location)

                        scene.node_tree.nodes['Image'].image = bg_img

                        # iterate over the keyframes and render
                        # LOOP TO RENDER
                        dict_info['bg'] = os.path.splitext(os.path.basename(background))[0]
                        dict_info['cloth'] = cloth_img_name
                        dict_info['light'] = sh_coeffs

                        scene.render.use_antialiasing = False
                        dict_info['img_path'] = '%04d-%s-%04d-%s-%s-%d.png' % (img_ct, name.replace(" ", ""), get_real_frame(seq_frame), gndr, os.path.splitext(os.path.basename(background))[0], orientation)
                        scene.render.filepath = join(output_path, dict_info['img_path'])

                        log_message("Rendering image %d: frame %d with: gender %s, shape %d, background %s, orientation %d" % (img_ct, seq_frame, gndr, ishape, os.path.splitext(os.path.basename(background))[0], orientation))
            
                        controller = mute()
                        # Render
                        start_timer = time.time()
                        bpy.ops.render.render(write_still=True)
                        unmute(controller)
                        print("WOOOOW, it takes %f s to render"%(time.time() - start_timer))

                        # bone locations should be saved after rendering so that the bones are updated
                        bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
                        dict_info['joints2D'] = np.transpose(bone_locs_2D)
                        dict_info['joints3D'] = np.transpose(bone_locs_3D)

                        reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
                        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

                        img_ct += 1
                        sequence_info.append(dict_info)

        bpy.ops.wm.read_homefile()
        log_message("Completed batch")
    
    # save annotation excluding png/exr data to _info.json file
    with open(jsonfile_info, 'w', encoding='utf-8') as f:
        json.dump(sequence_info, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    main()
