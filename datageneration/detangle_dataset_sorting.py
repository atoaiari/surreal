import os
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def detangle_dataset_sorting(dataset_path, factors, tt_plit):
    info_dir = os.path.join(dataset_path, "info")
    info_files = os.listdir(info_dir)
    
    dfs = []
    for file in info_files:
        with open(os.path.join(info_dir, file)) as f:
            json_data = pd.json_normalize(json.loads(f.read()))
        dfs.append(json_data)
    df = pd.concat(dfs, sort=False)

    df["pose_id"] = df["index"].astype(str).str.zfill(4) + "-" + df["frame"].astype(str).str.zfill(5)
    df.drop(columns=["camDist", "shape", "pose", "joints2D", "joints3D", "light", "zrot", "camLoc", "index", "frame", "img_idx", "cloth"], inplace=True)
    
    df.sort_values(by=factors, ignore_index=True, inplace=True)
    factor_sizes = [len(df[factor].unique()) for factor in factors]
    factor_bases = np.prod(factor_sizes) / np.cumprod(factor_sizes)
    print(f"Factor Sizes: {factor_sizes}")
    print(f"Factor Bases: {factor_bases}")
    print(f"Tot imgs: {df.shape[0]}")

    info_dict = df.to_dict('records')
    info_dict = {"created": datetime.now().strftime("%d-%m-%Y-%H-%M"), "factors": factors, "factor_bases": factor_bases, "factor_sizes": factor_sizes, "data": info_dict}
    with open(os.path.join(dataset_path, f"detangle_surreal.json"), 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # dataset = {}
    # dataset["train"], dataset["test"] = train_test_split(df, test_size=tt_split)
    # for split_type in ["train", "test"]:
    #     dataset[split_type].sort_values(by=factors, ignore_index=True, inplace=True)
    #     print(f"Dataset {split_type}")
    #     factor_sizes = [len(dataset[split_type][factor].unique()) for factor in factors]
    #     factor_bases = np.prod(factor_sizes) / np.cumprod(factor_sizes)
    #     print(f"Factor Sizes: {factor_sizes}")
    #     print(f"Factor Bases: {factor_bases}")
    #     print(f"Tot imgs: {dataset[split_type].shape[0]}")
    #     info_dict = dataset[split_type].to_dict('records')
    #     with open(os.path.join(dataset_dir, f"detangle_surreal_{split_type}.json"), 'w', encoding='utf-8') as f:
    #         json.dump(info_dict, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    print("Done!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="path to the dataset")
    parser.add_argument("--factors", type=str, nargs="+", default=["pose_id", "gender", "orientation", "bg"], help="sorted list of factors to consider")
    parser.add_argument("--split", type=float, default=0, help="train-test split")
    args = parser.parse_args()

    detangle_dataset_sorting(args.dataset_path, args.factors, args.split)