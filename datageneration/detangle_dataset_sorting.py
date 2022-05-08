import os
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
from collections import OrderedDict
import itertools

from utils.utils import *


def detangle_dataset_sorting(dataset_path, factors_dict, tt_split):
    info_dir = os.path.join(dataset_path, "info")
    info_files = os.listdir(info_dir)

    dfs = []
    for file in info_files:
        print(file)
        with open(os.path.join(info_dir, file)) as f:
            json_data = pd.json_normalize(json.loads(f.read()))
        dfs.append(json_data)
    df = pd.concat(dfs, sort=False)

    with open(factors_dict, "r") as f:
        factors = json.load(f, object_pairs_hook=OrderedDict)
    factor_keys = list(factors.keys())

    df["pose_id"] = df["index"].astype(str).str.zfill(4) + "-" + df["frame"].astype(str).str.zfill(5)
    df.drop(columns=["camDist", "shape", "pose", "joints2D", "joints3D", "light", "zrot", "camLoc", "index", "frame", "img_idx", "cloth"], inplace=True)
    
    for key, value in factors.items():
        if key in df.columns and value > 0:
            if key == "orientation":
                assert value in [4, 8, 16]
                orientations = np.arange(0, 360, (360/value))
                df = df[df[key].isin(orientations)]
            else:
                possible_values = df[key].unique()
                df = df[df[key].isin(possible_values[:value])]

    df.sort_values(by=factor_keys, ignore_index=True, inplace=True)
    factor_sizes = [len(df[factor].unique()) for factor in factor_keys]
    factor_bases = np.prod(factor_sizes) / np.cumprod(factor_sizes)
    print(f"Factor Sizes: {factor_sizes}")
    print(f"Factor Bases: {factor_bases}")
    print(f"Tot imgs: {df.shape[0]}")

    info_dict = {
        "created": datetime.now().strftime("%d-%m-%Y-%H-%M"), 
        "factors": factor_keys, 
        "factor_bases": factor_bases, 
        "factor_sizes": factor_sizes, 
        "n_samples": df.shape[0], 
        "images": df.to_dict('records'),
        "features": [list(i) for i in itertools.product(*[range(x) for x in factor_sizes])]}

    output_file = "detangle_surreal"
    for key, value in factors.items():
        output_file += f"_{key.split('_')[0]}-{len(df[key].unique())}"
    output_file += ".json"

    os.makedirs(os.path.join(dataset_path, "dataset"), exist_ok=True)
    with open(os.path.join(dataset_path, "dataset", output_file), 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    print("Done!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="path to the dataset")
    # parser.add_argument("--factors", type=str, nargs="+", default=["pose_id", "gender", "orientation", "bg", "shape_idx"], help="sorted list of factors to consider")
    parser.add_argument("--factors", type=str, default="configs/surreal_factors.json", help="path to the json containing the list of factors")
    parser.add_argument("--split", type=float, default=0, help="train-test split")
    args = parser.parse_args()

    detangle_dataset_sorting(args.dataset_path, args.factors, args.split)