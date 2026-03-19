if __name__ == '__main__':
    import sys,os; cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(cur_dir, '..')))
    if __package__ is None \
        or __package__ == '': # 4debug
        __package__ = os.path.basename(  cur_dir  ); print(f'{__package__ =}')
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
import globals_
from . import aug
from conf2 import *


PRE_COMPUTE_IDXS:bool= True  #pre: load or gen in DatasetGTA.__init__. else compute in __getitem__
LOAD_IDXS_in_init_or_getitem: bool= False  # 1=init,0=getitem. valid only when PRE_COMPUTE_IDXS
PRE_COMPUTE_IDXS = 0 # for humanise, compute in getitem is enough (8 workers, CPU usage 10+%

def get_idxs_of_near_point_in_scene_vert( # filtered points too far from human 
    scene_vert,root_joint,
    thres,
    GPU_accelerate:bool=False,
):
    if GPU_accelerate  :    
        root_joint=torch.tensor(root_joint).cuda()
        dist = torch.norm(scene_vert[:,:3] - root_joint, dim=-1)
        idxs = torch.where(dist <= thres)[0].cpu().numpy()
    else:
        dist = np.linalg.norm(scene_vert[:,:3] - root_joint, axis=-1)
        idxs = np.where(dist <= thres)[0]
    return idxs

class DatasetHumanise(Dataset):
    def __init__(self,mode,dataset_specs={},scene_seen_or_unseen:str=None) -> None:
        IDXS_parent_DIR=HUMANISE_DIR/'mid_2_idxs-Nov01'
        SPLIT_FILE=HUMANISE_DIR/'split'/f'{SPLIT_FILE_NAME}.json'
        self.HUHUMANISE_DIR=HUMANISE_DIR
        self.J_S_DIR=J_S_DIR
        assert J_S_DIR.exists(),   J_S_DIR
        
        
        self.mode = mode
        self.t_his = t_his 
        self.t_pred = t_pred 
        self.t_total = t_his + t_pred
        self.random_rot = dataset_specs.get('random_rot',False)
        self.is_contact = dataset_specs.get('is_contact',False)
        self.is_frame_contact = dataset_specs.get('is_frame_contact',False)
        self.step = step = STEP
        self.num_scene_points = num_scene_points = MAX_OBJ_PC_NUMBER
        self.max_dist_from_human = max_dist_from_human = dataset_specs.get('max_dist_from_human', 2.5)
        self.wscene = wscene = dataset_specs.get('wscene',True)
        self.wcont = wcont = dataset_specs.get('wcont',True)
        self.num_cont_points = num_cont_points = dataset_specs.get('num_cont_points', 500)
        
        
        annotations = pd.read_csv(os.path.join(  J_S_DIR/ 'annotation.csv'))
        total_amount = annotations.shape[0]
        with open(SPLIT_FILE,'r') as f:
            """
            splitB structure:
            {
                'train':trainset_motion_paths,
                'test':{
                    'scene_seen':scene_seen_motion_paths,
                    'scene_unseen':scene_unseen_motion_paths,
                }
            }
            """
            mode_2_motion_path=json.load( f)
            motion_paths=mode_2_motion_path[mode]
            del mode_2_motion_path
            if mode=='test':
                if scene_seen_or_unseen is   None:
                    motion_paths=motion_paths['scene_seen']+motion_paths['scene_unseen']
                else:
                    scene_seen_or_unseen=scene_seen_or_unseen.lower()
                    if scene_seen_or_unseen=='seen':
                        motion_paths=motion_paths['scene_seen']
                    elif scene_seen_or_unseen=='unseen':
                        motion_paths=motion_paths['scene_unseen']
                    else:
                        raise ValueError(f"Invalid value for scene_seen_or_unseen: {scene_seen_or_unseen}")
        if   0  :  
            print('warning  for debug')
            motion_paths= motion_paths[:2]
        print('loading motion...')
        mid2m={} # m: motion: joints
        for motion_path in tqdm(motion_paths, mininterval=5):
            motion_path = Path( motion_path )
            motion_name = motion_path.name
            mid = int( motion_path.stem )
            motion_path = J_S_DIR/motion_name
            # print(f"{motion_path=}")
            joints=np.load(motion_path)
            if   0  :# the motions paths read from split json is filtered version
                MIN_num_frames= t_total
                if len(joints)<MIN_num_frames:
                    continue
            mid2m[mid]=joints
        print(f"{len(motion_paths)=}",  )
        del total_amount
        #--------------filter over-----------------------------
        sid2s={} #scene id 2 scene
        mid2sid={}
        # for motion_path in tqdm(motion_paths, mininterval=5):
        for motion_path in motion_paths:
            mid = int(motion_path.split('/')[-1].split('.')[0]) 
            sid = annotations.loc[mid]['scene_id'] #scene_id
            #
            mid2sid[mid] = sid
            sid2s[sid]=None
        print('loading scene...')
        for sid in tqdm(sid2s, mininterval=5):
            spath=S_DIR/f"{sid}.xyz"
            scene=np.loadtxt(spath,skiprows=1)
            scene=scene[:,:3]
            assert scene.shape[1]==3,(  spath,  scene.shape)
            sid2s[sid]=scene
        
        self.mid2m = mid2m
        self.mid2sid = mid2sid
        self.sid2s = sid2s
        #---------------------------------
        self.data = {}
        self.data_item_index_2_near_whole_objs_id = {}#whole: the whole obj (all points)
        self.data_item_index_2_near_partial_objs_id = {}#partial: only points of the part within max_dist_from_human
        self.data_item_index_2_max_dist = {}   
        k = 0
        
        
        #---------------------------------
        self.data = {} #data item index 2 f'{mid}.{frame index}'
        self.scene_point_idx = {}
        k = 0
        #  motion index/ motion_id/mid. 'sub' in DatasetGTA
        for im,sid in tqdm(mid2sid.items(), mininterval=5):
            seq_len = mid2m[im].shape[0]
            idxs_frame = np.arange(0,seq_len - self.t_total + 1,step)
            for tmp_i,i in enumerate(idxs_frame):
                self.data[k] = f'{im}.{i}'
                k += 1
        print(f't_total {self.t_total}.  {k} data items')  
        if  0  :  
            print('warning  for debug2')
            _N_4_debug = 13
            _keys = list(self.data.keys())[:_N_4_debug]
            self.data = {k: self.data[k] for k in _keys}
            del _keys, _N_4_debug
        self.id2O=[  1 for _ in range(  len(list(self.data.keys()))  )  ]
        if PRE_COMPUTE_IDXS:
            #-----modify from humanise----------------
            _str_idx=f"{S_DIR.stem=}-{SPLIT_FILE_NAME=}-t_his{t_his}t_pred{t_pred}step{step}--{mode}"
            if mode=='test' and scene_seen_or_unseen is not None:
                _str_idx+='-'
                _str_idx+=scene_seen_or_unseen
            IDXS_DIR= IDXS_parent_DIR/ _str_idx
            self.IDXS_DIR=IDXS_DIR
            print(f"IDXS_DIR: {str(IDXS_DIR)}")
            #
            print('we assume that cache is created if folder exist , then load_idx=1')
            load_idx= IDXS_DIR.exists()  
            # load_idx= 0
            if load_idx and not LOAD_IDXS_in_init_or_getitem:
                print(f'IDXS_DIR already created && load in getitem => return')
                return
            #
            IDXS_DIR.mkdir(exist_ok=True)
            print(f'{load_idx=}',)
            k=0
            for im,sid in tqdm(mid2sid.items(), mininterval=5):
                seq_len = mid2m[im].shape[0]
                idxs_frame = np.arange(0,seq_len - t_total + 1,step)
                scene_vert = sid2s[sid]
                    
                GPU_accelerate=   1
                if GPU_accelerate:
                    scene_vert=torch.tensor(scene_vert).cuda()
                
                
                for tmp_i,i in enumerate(idxs_frame):
                    pose = mid2m[im][i:t_total+i]
                    root_joint = pose[t_his-1, ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
                    idxs = get_idxs_of_near_point_in_scene_vert(scene_vert,root_joint,self.max_dist_from_human,GPU_accelerate)
                    if 1:
                        assert len(idxs)>0,(im,tmp_i,i)
                    if 1:
                        if len(idxs)<1000:
                            print(f"==>> im: {im}",end=' ')
                            print(f"==>> len(idxs): {len(idxs)}")
                            
                    self.scene_point_idx[k] = idxs.astype(np.int32)
                    if not load_idx:
                        save_path = IDXS_DIR/f"{k}.pkl"
                        if save_path.exists():
                            print(f"warning: {save_path} exists")
                        with open(save_path, 'wb') as f:
                            pickle.dump(idxs, f)
                    k+=1
            print(f'seq length {t_total},in total {k} seqs')        
        
    def __len__(self):
        return len(list(self.data.keys()))


    def __getitem__(self, idx):
        """
        idx is k in init
        """

        item_key = self.data[idx].split('.')
        sub = mid = int(item_key[0])
        fidx = int(item_key[1])

        pose = torch.tensor(self.mid2m[mid][fidx:fidx + t_total]).float()

        sid=self.mid2sid[mid]
        scene_vert = self.sid2s[sid]
        scene_vert = torch.tensor(scene_vert).float()
        
        
        if PRE_COMPUTE_IDXS:
            if LOAD_IDXS_in_init_or_getitem:
                idxs = self.scene_point_idx[idx]
            else:# read idxs from disk
                IDXS_DIR= self.IDXS_DIR
                idxs_path = IDXS_DIR/f"{idx}.pkl"
                with open(idxs_path, 'rb') as f:
                    idxs = pickle.load(f)
        else:
            root_joint = pose[t_his-1, ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
            idxs = get_idxs_of_near_point_in_scene_vert(scene_vert.numpy(),root_joint.numpy(),self.max_dist_from_human,)
            if 1:
                assert len(idxs)>0,(idx,)
            if 1:
                if len(idxs)<1000:
                    print(f"==>> idx: {idx}",end=' ')
            idxs = idxs.astype(np.int32)
            del root_joint
        len_vidx = len(idxs)
        if len_vidx < self.num_scene_points:
            ids = list(range(len_vidx)) + np.random.choice(np.arange(len_vidx),self.num_scene_points-len_vidx).tolist()
        else:
            ids = np.random.choice(np.arange(len_vidx), self.num_scene_points, replace=False)
        v_idx = idxs[ids]
        scene_vert = scene_vert[v_idx]
        
        if AUG and globals_.TRAIN:
            scene_vert, pose = aug.A(scene_vert, pose)
        
        scene_origin = torch.clone(pose[t_his-1,ROOT_JOINT_IDX:ROOT_JOINT_IDX+1])#last observed root joint as scene origin
        
        
        scene_vert = scene_vert - scene_origin # [5000, 3]  N,3
        pose = pose - scene_origin # [90, 21, 3]  T,J,3

        if  0   :
            objs_pc=0
        else:
            objs_pc=scene_vert
            objs_pc=objs_pc.view(1,MAX_OBJ_PC_NUMBER,3)
        if 1:
            objs_semId=torch.zeros((1,))
        
        print_randomly(f"objs_pc: {objs_pc.shape}", 
            0.0001
        )
        return pose, objs_pc, scene_origin, objs_semId, item_key
