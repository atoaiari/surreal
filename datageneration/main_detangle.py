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
import configs.config as config

from utils.utils import *
from utils.blender_utils import *

sys.path.insert(0, ".")

def main():
    # parse commandline arguments
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images for disentanglement.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--gender', type=str, choices=["female", "male"])
    parser.add_argument('path', type=str, help='basic config file path')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    
    idx = args.idx
    log_message("input idx: %d" % idx)
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

    #TODO: params and generation_config can be integrated in a current_configuration_file
    params = config.load_file(args.path, 'SYNTH_DATA')
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    stepsize = params['stepsize']

    # loading generation config
    try:
        with open(os.path.join(output_path, "generation_config.json")) as f :
            generation_config = json.load(f)
    except IOError: # parent of IOError, OSError *and* WindowsError where available
        print("No generation config file found!")
        exit(1)
    
    # name is set given idx
    name = idx_info['name']
    info_path = generation_config["info_path"]
    params['info_path'] = info_path
    logs_path = generation_config["logs_path"]
    params["logs_path"] = logs_path
    images_path = generation_config["images_path"]
    params["images_path"] = images_path
    sequence_output_path = join(images_path, '%s' % name.replace(" ", ""))
    params['sequence_output_path'] = sequence_output_path
    tmp_path = join(tmp_path, '%s' % (name.replace(" ", "")))
    params['tmp_path'] = tmp_path

    # check if already computed
    #  + clean up existing tmp folders if any
    if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
        os.system('rm -rf %s' % tmp_path)   

    # create tmp directory
    if not exists(tmp_path):
        mkdir_safe(tmp_path)

    log_message("log path: %s/%s.log"%(logs_path, name.replace(" ", "")))
    import logging 
    from imp import reload
    reload(logging)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("%s/%s.log"%(logs_path, name.replace(" ", "")), mode='a', encoding='utf-8', delay=False)
        ]
    )

    jsonfile_info = join(info_path, "%04d-"%idx + name.replace(" ", "") + "-%s"%args.gender + "-info.json")
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

    # factors
    factors_config = generation_config["factors"]

    # frames per sequence
    frames_per_sequence = factors_config["frames_per_sequence"]
    log_message("frames used from the sequence: %d" %frames_per_sequence)
    tot_frames = int(idx_info['nb_frames'] * stepsize)
    assert(frames_per_sequence < tot_frames)
    # select frames_per_sequence - 1 random frames (the first is always 0)
    sequence_info = []
    frames = list(np.random.choice(np.arange(1, tot_frames), size=frames_per_sequence-1, replace=False))
    frames.append(0)
    frames.sort()

    # gender
    # gender = factors_config["gender"]
    gender = [args.gender]        # only because of ram size

    # backgrounds
    backgrounds = factors_config["backgrounds"]
    log_message("number of backgrounds: %d" % len(backgrounds))
    
    # orientations
    orientations = factors_config["orientations"]
    log_message("number of orientations: %d" % len(orientations))

    # shapes
    shapes = factors_config["shapes"]
    log_message("number of shapes: %d" % len(shapes))

    # clothing textures
    textures = factors_config["textures"]
    log_message("number of textures: %d" % len(textures))
    
    # random light
    sh_coeffs = generation_config["sh_coeffs"]

    # camera distance
    camera_distance = generation_config["camera_distance"]
    log_message("camera distance: %.2f"%camera_distance)
    
    img_ct = 0
    for igndr, gndr in enumerate(gender):
        for itexture, texture in enumerate(textures):
            ### BLENDER ###
            scene = bpy.data.scenes['Scene']
            scene.render.engine = 'CYCLES'

            bpy.data.materials['Material'].use_nodes = True
            scene.cycles.shading_system = True
            scene.use_nodes = True
            
            cloth_img = bpy.data.images.load(join(smpl_data_folder, texture))
            log_message("clothing texture: %s" % texture)
            
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
            
            data = cmu_parms[name]

            # for each clipsize'th frame in the sequence
            get_real_frame = lambda ifr: ifr
            random_zrot = 0
            # additional z rotation, not used
            # random_zrot = 2*np.pi*np.random.rand()
            # shapes = [shapes[1]]

            arm_ob.animation_data_clear()
            cam_ob.animation_data_clear()    

            for part, material in materials.items():
                material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

            for ish, coeff in enumerate(sh_coeffs):
                for sc in scs:
                    sc.inputs[ish+1].default_value = coeff

            for ishape, shape in enumerate(shapes):
                reset_loc = True
                # curr_shape = reset_joint_positions(orig_trans, curr_shape, ob, arm_ob, obname, scene,
                #                                     cam_ob, smpl_data['regression_verts'], smpl_data['joint_regressor'])

                # create a keyframe animation with pose, translation, blendshapes and camera motion
                for iorientation, orientation in enumerate(orientations):
                    for background in backgrounds:        
                        bg_img = bpy.data.images.load(background)
                        res_paths = create_composite_nodes(scene.node_tree, params, img=bg_img, idx=idx)
                        
                        # i use the same translation for all the frames
                        translations = np.tile(data['trans'][0], (frames_per_sequence, 1))
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

                            dict_info['shape'] = shape
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
                            dict_info['cloth'] = texture
                            dict_info['light'] = sh_coeffs

                            scene.render.use_antialiasing = False
                            dict_info['img_path'] = '%04d-%s-f%04d-%s-%s-ori%d-sh%d-tex%d.png' % (img_ct, name.replace(" ", ""), get_real_frame(seq_frame), gndr, os.path.splitext(os.path.basename(background))[0], iorientation, ishape, itexture)
                            scene.render.filepath = join(sequence_output_path, dict_info['img_path'])
                
                            controller = mute()
                            # Render
                            start_rendering_timer = time.time()
                            bpy.ops.render.render(write_still=True)
                            unmute(controller)
                            rendering_time = time.time() - start_rendering_timer
                            log_message("Rendering image %4d for sequence %s: frame %4d with gender %s, shape %d, background %s, orientation %3.2f, texture %s - rendering time: %.2f" % (img_ct, name.replace(" ", ""), seq_frame, gndr, ishape, os.path.splitext(os.path.basename(background))[0], orientation, os.path.splitext(os.path.basename(texture))[0], rendering_time))

                            # bone locations should be saved after rendering so that the bones are updated
                            bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
                            dict_info['joints2D'] = np.transpose(bone_locs_2D)
                            dict_info['joints3D'] = np.transpose(bone_locs_3D)

                            reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
                            arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

                            img_ct += 1
                            sequence_info.append(dict_info)
                        

            bpy.ops.wm.read_factory_settings()
            bpy.ops.wm.read_homefile()
        log_message("Completed batch")
            
    # save annotation excluding png/exr data to _info.json file
    with open(jsonfile_info, 'w', encoding='utf-8') as f:
        json.dump(sequence_info, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    main()
