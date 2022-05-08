import json
import argparse
import sys
import configs.config as config
import os
import numpy as np
import random
from datetime import datetime
from utils.utils import *

def main():
    parser = argparse.ArgumentParser(description="Generate synth dataset images for disentanglement.")
    parser.add_argument("--frames", type=int, help="frames to use from the sequence", default=2)
    parser.add_argument("--gender", type=int,
                        help="-1: both, 0: female, 1: male", default=-1)
    parser.add_argument("--backgrounds", type=int,
                        help="number of backgrounds", default=10)
    parser.add_argument("--orientations", type=int, choices=[4, 8, 16], default=4,
                        help="number of orientation classes")
    parser.add_argument("--shapes", type=int, default=4,
                        help="number of shapes")
    parser.add_argument("--textures", type=int, default=8,
                        help="number of textures")
    parser.add_argument("--reset", action="store_true", help="reset the generation config file, even if it already exists")
    parser.add_argument("path", help="basic config path")
    args = parser.parse_args()

    configuration_dict = {}
    params = config.load_file(args.path, "SYNTH_DATA")

    if not os.path.isfile(os.path.join(params["output_path"], "generation_config.json")) or args.reset:
        seed_number = 11
        random.seed(seed_number)
        np.random.seed(seed_number)

        configuration_dict.update(params)
        configuration_dict["created"] = datetime.now().strftime("%d-%m-%Y-%H-%M")
        configuration_dict["factors"] = {"frames_per_sequence": args.frames}

        # backgrounds
        bg_names = os.path.join(params["bg_path"], 'train_img.txt')
        nh_txt_paths = []
        with open(bg_names) as f:
            for line in f:
                nh_txt_paths.append(os.path.join(params["bg_path"], line[:-1]))
        # backgrounds = np.random.choice(nh_txt_paths[:-1], args.backgrounds, replace=False)    
        backgrounds = nh_txt_paths[:args.backgrounds]
        configuration_dict["factors"]["backgrounds"] = backgrounds

        # gender
        genders = {0: 'female', 1: 'male'}
        # set gender.
        if args.gender == -1:
            gender = [genders.get(g) for g in genders]
        else:
            gender = genders.get(args.gender)
        configuration_dict["factors"]["gender"] = gender

        # orientations
        configuration_dict["factors"]["orientations"] = list(np.arange(0, 360, (360/args.orientations)))

        # clothing/textures
        assert args.textures % 2 == 0
        textures = []
        for igndr, gndr in enumerate(gender):
            with open(os.path.join(params["smpl_data_folder"], 'textures', '%s_%s.txt' % (gndr, 'train'))) as f:
                txt_paths = f.read().splitlines()
            # if using only one source of clothing
            if params["clothing_option"] == 'nongrey':
                clothing_txt_paths = [k for k in txt_paths if 'nongrey' in k]
            elif params["clothing_option"] == 'grey':
                clothing_txt_paths = [k for k in txt_paths if 'nongrey' not in k]

            textures.extend(np.random.choice(clothing_txt_paths, size=int(args.textures / 2), replace=False))
        configuration_dict["factors"]["textures"] = textures

        # shapes (extracted only from female model)
        ndofs = 10
        gndr = "female"
        smpl_data = np.load(os.path.join(params["smpl_data_folder"], params["smpl_data_filename"]))
        fshapes = smpl_data['%sshapes' % gndr][:, :ndofs]
        nb_fshapes = len(fshapes)
        fshapes = fshapes[:int(nb_fshapes*0.8)]     # train split
        shapes_idx = np.random.choice(np.arange(len(fshapes)), size=args.shapes, replace=False)
        shapes = fshapes[shapes_idx]
        configuration_dict["factors"]["shapes"] = shapes

        # light
        configuration_dict["sh_coeffs"] = .7 * (2 * np.random.rand(9) - 1)
        configuration_dict["sh_coeffs"][0] = .5 + .9 * np.random.rand() # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
        configuration_dict["sh_coeffs"][1] = -.7 * np.random.rand()

        # camera distance
        # configuration_dict["camera_distance"] = np.random.normal(8.0, 1)
        configuration_dict["camera_distance"] = 7.2   # fixed not random

        if args.reset and os.path.exists(params["output_path"]) and params["output_path"] != "" and params["output_path"] != "/":
            os.system(f"rm -rf {params['output_path']}")
        os.makedirs(params["output_path"], exist_ok=True)
        
        folders = ["info", "images", "logs", "dataset"]   
        for folder in folders:
            os.makedirs(os.path.join(params["output_path"], folder), exist_ok=True)
            configuration_dict[f"{folder}_path"] = str(os.path.join(params["output_path"], folder))

        with open(os.path.join(params["output_path"], "generation_config.json"), "w", encoding="utf-8") as f:
            json.dump(configuration_dict, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        
        print("Generated a new configuration file!")
    else:
        print("Configuration file already exists!")
    

if __name__ == "__main__":
    main()