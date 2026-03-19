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

class DatasetGTA(Dataset):

    def __init__(self, mode, scene_seen_or_unseen='wont be used',dataset_specs={}):
        J_S_DIR=gta_data_path
        assert J_S_DIR.exists(),   J_S_DIR
        
        self.mode = mode
        self.t_his = t_his 
        self.t_pred = t_pred 
        self.t_total = t_his + t_pred
        self.random_rot = dataset_specs.get('random_rot',False)
        self.is_contact = dataset_specs.get('is_contact',False)
        self.is_frame_contact = dataset_specs.get('is_frame_contact',False)
        self.step = step = (STEP if mode=='train' else 5)
        self.num_scene_points = num_scene_points = MAX_OBJ_PC_NUMBER
        self.max_dist_from_human = max_dist_from_human = dataset_specs.get('max_dist_from_human', 2.5)
        self.wscene = wscene = dataset_specs.get('wscene',True)
        self.wcont = wcont = dataset_specs.get('wcont',True)
        self.num_cont_points = num_cont_points = dataset_specs.get('num_cont_points', 500)
        self.sigma = dataset_specs.get('sigma', 0.02)
        self.cont_thre = 0.2
        
        
        self.scene_split = {'train': ['r001','r002','r003', 'r006'],
                            'test': ['r010', 'r011', 'r013']
                            }
        
        
        # motion_paths_all = natsorted(glob.glob(  str(J_S_DIR/'*.npy')   )) 
        motion_paths_all=list(J_S_DIR.glob('*_sf*.npy'))  #all: before split
        motion_paths=[]
        for motion_path in motion_paths_all:
            name_withoutSuffix=str(  motion_path.stem  ) # eg. 2020-06-04-23-08-16_r013_sf0
            # if '2020-06-04-22-57-20_r013_sf0' not in seq:
            #     continue
            room = name_withoutSuffix.split('_')[1]#eg. r013
            if room not in self.scene_split[mode]:
                continue
            motion_paths.append(motion_path)
        if   0  :  
            print('warning  for debug')
            motion_paths= motion_paths[:  13 ]
        print(  f"{len(motion_paths)=}")
            
            
        mid2m={} # m: motion: joints
        print('loading motions...')
        for motion_path in tqdm(motion_paths, mininterval=5):
            joints=np.load(motion_path,MMAP_MODE__gta_joint)
            name_withoutSuffix=str(  motion_path.stem  )
            mid = name_withoutSuffix
            assert joints.shape[0]<=2000
            assert joints.shape[1]==21
            assert joints.shape[2]==3
            mid2m[mid]=joints
        sid2s={} #scene id 2 scene.   sid:  2020-06-04-22-57-20_r013
        mid2sid={}
        for motion_path in motion_paths:
            name_withoutSuffix=str(  motion_path.stem  )
            mid = name_withoutSuffix
            sid = name_withoutSuffix.split('_sf')[0] #scene_id
            #
            mid2sid[mid] = sid
            sid2s[sid]=None
        print('loading scenes...')
        for sid in tqdm(sid2s, mininterval=5):
            spath=J_S_DIR/f"{sid}.npy"
            scene=np.load(spath,mmap_mode=MMAP_MODE__gta_scene)
            sid2s[sid]=scene

        self.mid2m = mid2m
        self.mid2sid = mid2sid
        self.sid2s = sid2s
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

        
        self.id2O=[  1 for _ in range(  len(list(self.data.keys()))  )  ]
        if PRE_COMPUTE_IDXS:
            scene_seen_or_unseen=None
            IDXS_parent_DIR=gta_idx_path
            #-----modify from humanise----------------
            _str_idx=f"A-t_his{t_his}step{step}--{mode}"
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
                    
                # if load_idx:
                #     save_path = os.path.join( IDXS_DIR/f"{im}.pkl"   )
                #     with open(save_path, 'rb') as f:
                #         sub_idxs_list=pickle.load( f)
                #     assert len(idxs_frame)==len(sub_idxs_list),f"{len(idxs_frame)=} {len(sub_idxs_list)=}"
                # else:
                GPU_accelerate=   1
                if GPU_accelerate:
                    scene_vert=torch.tensor(scene_vert).cuda()
                
                
                for tmp_i,i in enumerate(idxs_frame):
                    pose = mid2m[im][i:t_total+i]
                    root_joint = pose[t_his-1, 14:15]
                        
                    # if load_idx:
                    #     idxs=sub_idxs_list[tmp_i]   
                    # else:
                    if GPU_accelerate  :    
                        root_joint=torch.tensor(root_joint).cuda()
                        dist = torch.norm(scene_vert[:,:3] - root_joint, dim=-1)
                        idxs = torch.where(dist <= self.max_dist_from_human)[0].cpu().numpy()
                    else:
                        dist = np.linalg.norm(scene_vert[:,:3] - root_joint, axis=-1)
                        idxs = np.where(dist <= self.max_dist_from_human)[0]# filtered points too far from human
                    
                    
                    if 1:
                        assert len(idxs)>0,(im,tmp_i,i)
                    if 1:
                        if len(idxs)<1000:
                            print(f"==>> im: {im}",end=' ')
                            print(f"==>> len(idxs): {len(idxs)}")
                            
                    self.scene_point_idx[k] = idxs.astype(np.int32)
                    # different from hunanmise, for gta here is perFrame (per sub motion),  not per motion
                    if not load_idx:
                        save_path = IDXS_DIR/f"{k}.pkl"
                        if save_path.exists():
                            print(f"warning: {save_path} exists")
                        with open(save_path, 'wb') as f:
                            pickle.dump(idxs, f)
                    k+=1
                    

    def __len__(self):
        return len(list(self.data.keys()))


    def __getitem__(self, idx):
        """
        idx is k in init
        """

        item_key = self.data[idx].split('.')
        mid = item_key[0]
        fidx = int(item_key[1])# frame index

        pose = torch.tensor(self.mid2m[mid][fidx:fidx + self.t_total]).float()

        sid=self.mid2sid[mid]
        scene_vert = self.sid2s[sid]
        scene_vert = torch.tensor(scene_vert).float()
        
        

        if LOAD_IDXS_in_init_or_getitem:
            idxs = self.scene_point_idx[idx]
        else:# read idxs from disk
            IDXS_DIR= self.IDXS_DIR
            room= sid.split('_')[1]
            idxs_path = IDXS_DIR/f"{idx}.pkl"
            with open(idxs_path, 'rb') as f:
                idxs = pickle.load(f)
        len_vidx = len(idxs)
        if len_vidx < self.num_scene_points:
            ids = list(range(len_vidx)) + np.random.choice(np.arange(len_vidx),self.num_scene_points-len_vidx).tolist()
        else:
            ids = np.random.choice(np.arange(len_vidx), self.num_scene_points, replace=False)
        v_idx = idxs[ids]
        scene_vert = scene_vert[v_idx]
        
        if AUG and globals_.TRAIN:
            scene_vert, pose = aug.A(scene_vert, pose)
        
        scene_origin = torch.clone(pose[t_his-1,14:15])#last hist frame root joint as new scene origin
        
        
        scene_vert = scene_vert - scene_origin # [5000, 3]  N,3
        pose = pose - scene_origin # [90, 21, 3]  T,J,3

        if  0   :
            objs_pc=0
        else:
            objs_pc=scene_vert
            objs_pc=objs_pc.view(1,MAX_OBJ_PC_NUMBER,3)
        if 1:
            objs_semId=torch.zeros((1,))
        
        return pose, objs_pc, scene_origin, objs_semId, item_key

    

