'''
the code to gen train/test split at `./data/humanise/split.json`, 

python datasets/dataset_preprocess/humanise/gen-split/gen_split.py
'''

# sys path add repo root (this script lives at datasets/dataset_preprocess/humanise/gen-split/)
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
sys.path.append(REPO_ROOT)
os.chdir(REPO_ROOT) # conf.py uses cwd-relative paths (eg. mkdir), so anchor cwd to repo root

import json
import utils
from conf import *
from pathlib import Path
import os
import pickle
import numpy as np
import torch
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from natsort import natsorted
from utils import *

class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MutualDistance.pkl') # seen/unseen partition pkl from MutualDistance author
ANNO_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotation__with__pure_motion_id.csv')

SPLIT_FILE=HUMANISE_DIR/'split.json' # output file
annotations = pd.read_csv(ANNO_CSV)

if __name__ == '__main__':

    total_amount = annotations.shape[0]
    id_s_m = annotations['pure_motion_id'].unique()
    scenes=annotations['scene_id'].unique()
    seed=   0
    print(f"---seed: {seed}")
    np.random.seed(seed)
    np.random.shuffle(id_s_m)
    # trainset_id_s_m = []#paste here
    if 1:
        """"
        4 fields:
            train
            seen_scene_unseen_motion
            seen_motion_unseen_scene
            unseen_motion_scene
        """
        with open(PKL, 'rb') as f:
            data = pickle.load(f)
        for key in data.keys():
            print (key, len(data[key]))
        trainset_id_s_m=data['train']
        unseen_m_seen_s__id_s_m=data['seen_scene_unseen_motion']
        unseen_m_unseen_s__id_s_m=data['unseen_motion_scene']
        seen_motion_unseen_scene__id_s_m=data['seen_motion_unseen_scene']
    #print len
    print(f"---{len(trainset_id_s_m)=}")
    print(f"---{len(unseen_m_seen_s__id_s_m)=}")
    print(f"---{len(unseen_m_unseen_s__id_s_m)=}")
    print(f"---{len(seen_motion_unseen_scene__id_s_m)=}")


    trainset_motion_paths = []
    # testset_motion_paths = []
    seen_scenes=set()
    scene_seen_motion_paths = []
    scene_unseen_motion_paths = []
    ct_missing=0
    ct_seen_motion_unseen_scene=0
    for _, anno in annotations.iterrows():
        pure_motion_id = anno['pure_motion_id']
        motion_id = anno['motion_id']
        scene_id = anno['scene_id']
        str_scene= scene_id[:-3]
        id_s_m= f"{str_scene}_{pure_motion_id}"
        motion_path=str(   J_S_DIR/f'{motion_id}.npy')
        if id_s_m in trainset_id_s_m:
            trainset_motion_paths.append( motion_path)
            seen_scenes.add(scene_id)
        elif  id_s_m in unseen_m_seen_s__id_s_m:
            scene_seen_motion_paths.append(  motion_path  )
        elif  id_s_m in unseen_m_unseen_s__id_s_m:
            scene_unseen_motion_paths.append( motion_path)
        else:
            if  id_s_m not in seen_motion_unseen_scene__id_s_m:
                ct_missing+=1
            else:
                ct_seen_motion_unseen_scene+=1
    print(f"{len(seen_scenes)=}")
    scenes=annotations['scene_id'].unique()
    unseen_scenes=set(scenes)-seen_scenes
    print(f"{len(unseen_scenes)=}")
    print(f'{ct_missing=}')
    print(f'{ct_seen_motion_unseen_scene=}')
    sum_=ct_seen_motion_unseen_scene+ct_missing+len(trainset_motion_paths)+len(scene_seen_motion_paths)+len(scene_unseen_motion_paths)
    print(f"==>> sum_: {sum_}")
    assert sum_==total_amount, f"{sum_=} != {total_amount=}"
    #  save mode_2_motion_path to SPLIT_FILE
    mode_2_motion_path={
        'train':trainset_motion_paths,
        'test':{
            'scene_seen':scene_seen_motion_paths,
            'scene_unseen':scene_unseen_motion_paths,
        }
    }



    SPLIT_FILE.parent.mkdir(parents=0,exist_ok=True)
    with open(SPLIT_FILE, 'w') as f:
        json.dump(mode_2_motion_path, f, separators=(',', ': '),indent=0)
    print(f"==>> SPLIT_FILE: {str(SPLIT_FILE)} saved")

