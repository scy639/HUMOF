if __name__ == '__main__':
    import sys,os; cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(cur_dir, '..')))
    if __package__ is None \
        or __package__ == '': # 4debug
        __package__ = os.path.basename(  cur_dir  ); print(f'{__package__ =}')
from hik.data.kitchen import Kitchen
import pickle
import numpy as np
import json
from my_py_lib.miscellaneous.DictWithTupleAsKey import DictWithTupleAsKey
import utils
from conf import *
from pathlib import Path
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pickle
import numpy as np
import torch
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted
from utils import *
import globals_
from . import aug

    

ACCE_filterB_by_cuda:bool =1 # whether 1 or 0, will be reset to 0 when __init__ ends
class DatasetHik(Dataset):
    """
    kid ie dataset_char ie A/B/C/D
    """
    def __init__(self,mode,dataset_specs={},scene_seen_or_unseen:str=None) -> None:
        self.ACCE_filterB_by_cuda = ACCE_filterB_by_cuda
        assert scene_seen_or_unseen is None
        self.HIK_preprocess_DIR=HIK_preprocess_DIR
        
        self.step = step = (STEP_train if mode=='train' else STEP_test)
        self.thres_filterB = thres_filterB = (THRES_filterB_train if mode=='train' else THRES_filterB)
        thres__primary_filterC = (THRES__primary_filterC_train if mode=='train' else THRES__primary_filterC_test)
        self.max_dist_from_human = max_dist_from_human = dataset_specs.get('max_dist_from_human', 3.5)
        self.max_dist_from_primary_to_other = max_dist_from_primary_to_other = 2.5
     
        if mode=='train':
            dataset_chars='ABC'
        elif mode=='test':
            dataset_chars='D'
        else:
            raise ValueError(f"Invalid mode: {mode}")
        kid2kitchen={}   # A/B/C/D  2  Kitchen
        for dataset_char in dataset_chars:
            kitchen = Kitchen.load_for_dataset(
                dataset=dataset_char, data_location=os.path.join(HIK_raw_DIR, "scenes")
            )
            kid2kitchen[dataset_char]=kitchen
        
        def load_pkl(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        kid__2__ip_2_iF2joints = {}
        kid__2__ip_2_iF2present = {}
        for dataset_char in dataset_chars:
            file_path= HIK_preprocess_DIR/f"hik_{dataset_char}"  /  "tus.pkl"
            print(f"{file_path=}")
            subseq,kid,present = load_pkl(file_path)
            ip_2_iF2joints=subseq
            print(f"{ip_2_iF2joints.shape=}")
            print(f"{present.shape=}")
            assert ip_2_iF2joints.shape[:2]== present.shape[:2]
            kid__2__ip_2_iF2joints [dataset_char] = ip_2_iF2joints
            kid__2__ip_2_iF2present [dataset_char] = present
        
        if 1:
            self.tu_2_others_ids = {}
            path_tu_2_others_ids = DIR_hik__tu_2_others_ids / (
                f'{DATASET_name}-{mode}-{scene_seen_or_unseen}-{step}-{t_his}-{t_total}'
                f'-{bool(ENABLE_filterB)}-{thres_filterB}.json'
                f'-{bool(PRIMARY_FILTER_C)}-{bool(PRIMARY_FILTER_C2)}-{thres__primary_filterC}.json'
            )  
            if path_tu_2_others_ids.exists():
                print(f'loading {path_tu_2_others_ids=}')
                # with open(path_tu_2_others_ids, 'r') as f:
                    # self.tu_2_others_ids = json.load(f)
                self.tu_2_others_ids = DictWithTupleAsKey.load( path_tu_2_others_ids )
                cache_missed__tu_2_others_ids = False
            else:
                print(f'not found {path_tu_2_others_ids=}, will compute it later')
                cache_missed__tu_2_others_ids = True
            # ---------
            self.tu_2_should_filter_primary = {}
            path_tu_2_should_filter_primary = DIR_hik__tu_2_should_filter_primary / (
                f'{DATASET_name}-{mode}-{scene_seen_or_unseen}-{step}-{t_his}-{t_total}'
                f'-{bool(PRIMARY_FILTER_C)}-{bool(PRIMARY_FILTER_C2)}-{thres__primary_filterC}.json'
            ) # implicit line joining
            if path_tu_2_should_filter_primary.exists():
                print(f'loading  {path_tu_2_should_filter_primary}')
                self.tu_2_should_filter_primary = DictWithTupleAsKey.load( path_tu_2_should_filter_primary )
                cache_missed__tu_2_should_filter_primary = False
            else:
                print(f'not found  {path_tu_2_should_filter_primary} , will compute it later')
                cache_missed__tu_2_should_filter_primary = True
            # ---------
            if not ( path_tu_2_others_ids.exists() and path_tu_2_should_filter_primary.exists() ) or PRIMARY_FILTER_C:
                if self.ACCE_filterB_by_cuda:
                    kid__2__ip_2_iF2joints = {k:torch.Tensor(v).cuda() for k,v in kid__2__ip_2_iF2joints.items()}
        self.kid2kitchen = kid2kitchen
        self.kid__2__ip_2_iF2joints = kid__2__ip_2_iF2joints
        self.kid__2__ip_2_iF2present = kid__2__ip_2_iF2present

        __ct_filtered_1 = 0
        __ct_filtered_2 = 0
        __ct_not_filtered = 0
        data = []
        id_2_numOth = []
        for dataset_char in dataset_chars:
            kid = dataset_char
            ip_2_iF2joints = kid__2__ip_2_iF2joints[dataset_char]
            ip_2_iF2present = kid__2__ip_2_iF2present[dataset_char]
            n_frames = ip_2_iF2joints.shape[1]
            for iF in tqdm(  range(0,n_frames-t_total-1,step  )   ):
                subseq = ip_2_iF2joints[:, iF:iF+t_total] # P,t_total,J,3
                present = ip_2_iF2present[:, iF:iF+t_total] # P,t_total
                n_p= len(  subseq )
                present_agg = np.any(present, axis=-1)  # (p)
                present_sum = np.sum(present, axis=-1)  # (p)
                for ip in range(n_p):
                    if not present_agg[ip]:
                        # print('not present_agg[ip] => skip')
                        __ct_filtered_1+=1
                        continue
                    # if present_sum[ip]<50:
                    if present_sum[ip] < t_total:
                        # print(f'{present_sum[ip]=}',end=' ')
                        # print_randomly(f'{present_sum[ip]=}', p=0.01)
                        __ct_filtered_2 += 1
                        continue
                    tu2 = (kid, iF, ip)
                    if PRIMARY_FILTER_C:
                        primary = self.kid__2__ip_2_iF2joints[kid][ip][iF:iF+t_total]  # t_total,J,3
                        if tu2 in self.tu_2_should_filter_primary:
                            assert not cache_missed__tu_2_should_filter_primary
                            should_filter_primary = self.tu_2_should_filter_primary[tu2]
                        else:
                            assert cache_missed__tu_2_should_filter_primary
                            should_filter_primary = primary_filterC( primary,thres__primary_filterC )
                            self.tu_2_should_filter_primary[tu2] = should_filter_primary
                        if should_filter_primary:
                            continue
                    len_others = self.get_others(kid, iF, ip, return_len=True )
                    if len_others == 0:
                        continue
                    __ct_not_filtered += 1
                    data.append(tu2)
                    id_2_numOth.append(len_others)
        print(f'{__ct_filtered_1=}')
        print(f'{__ct_filtered_2=}')
        print(f'{__ct_not_filtered=}')
        del __ct_filtered_1, __ct_filtered_2,__ct_not_filtered
        
        
        self.data=data
        self.id2O = id_2_numOth
        
        #---------------------------------
        print(f'{t_total=},{len(self.data)=}')        
        
        if   0  :
            id_2_numOth = []
            for i in range(len(self.data)):
                kid, iF, ip = self.data[i]
                len_others = self.get_others(kid, iF, ip, return_len=True )
                id_2_numOth.append(len(others))
            self.id2O = id_2_numOth
        print(f'{sys.getsizeof(self)/1024**3=:.2f} GB')
        if self.ACCE_filterB_by_cuda:
            self.kid__2__ip_2_iF2joints = {k:torch.Tensor(v).cpu().numpy() for k,v in self.kid__2__ip_2_iF2joints.items()}
            self.ACCE_filterB_by_cuda=False
        if 1:
            if cache_missed__tu_2_others_ids:
                # with open(path_tu_2_others_ids, 'w') as f:
                #     json.dump(self.tu_2_others_ids, f)
                DictWithTupleAsKey.dump( self.tu_2_others_ids, path_tu_2_others_ids )
                print(f'saved {path_tu_2_others_ids=}')
            if cache_missed__tu_2_should_filter_primary:
                DictWithTupleAsKey.dump( self.tu_2_should_filter_primary, path_tu_2_should_filter_primary )
                print(f'saved  {path_tu_2_should_filter_primary}')
    def get_others(self, kid, iF, ip, return_len:bool = False ) -> np.ndarray:
        # Check if the result is already cached
        tu = (kid, iF, ip)
        if   tu in self.tu_2_others_ids:
            others_ids = self.tu_2_others_ids[tu]
            if return_len:
                return len(others_ids)
            # if np
            # return torch.Tensor([self.kid__2__ip_2_iF2joints[kid][other_ip][iF:iF+t_his] for other_ip in others_ids]) #get warning(slow
            return torch.Tensor(  self.kid__2__ip_2_iF2joints[kid][others_ids][ :, iF:iF+t_his]  )
            # if Tensor
            # return torch.stack([self.kid__2__ip_2_iF2joints[kid][other_ip][iF:iF+t_his] for other_ip in others_ids])
        # kid, iF, ip = self.data[idx]
        primary = self.kid__2__ip_2_iF2joints[kid][ip][iF:iF+t_total]  # t_total,J,3
        root_primary = primary[t_his-1][ROOT_JOINT_IDX]  # 3
        if 1:
            others = []
            others_ids = []
            n_p=  len( self.kid__2__ip_2_iF2joints[kid] ) 
            for _i in range(  n_p   ):
                if _i == ip:
                    continue
                present = self.kid__2__ip_2_iF2present[kid][_i][iF:iF+t_total]  # t_total,
                assert present.shape==(t_total,)
                present_sum = np.sum(present, axis=-1)  # 1
                # if present_sum <  50:
                if present_sum  <  t_total:
                    continue
                other = self.kid__2__ip_2_iF2joints[kid][_i][iF:iF+t_total]  # t_total,J,3
                others.append(other)
                others_ids.append(_i)
            max_dist_from_primary_to_other = self.max_dist_from_primary_to_other
            while 1:
                if 1:
                    others_filtered = []
                    others_ids_filtered = []
                    for other_id, other in zip(others_ids, others):
                        if not ENABLE_filterB:
                            root_other = other[t_his-1][ROOT_JOINT_IDX]  # 3
                            dis_to_primary = np.linalg.norm( root_other-root_primary)
                            if dis_to_primary > max_dist_from_primary_to_other:
                                continue
                        else:
                            if filterB(other,primary,self.thres_filterB,self.ACCE_filterB_by_cuda):
                                continue
                        others_filtered.append(other[:t_his])
                        others_ids_filtered.append(other_id)
                n_oth = len(others_filtered)
                if n_oth >= min(MIN_OTHER_NUM, len(others)):
                    break
                max_dist_from_primary_to_other *= 1.3
            assert len(others_ids_filtered)==len(others_filtered)
            self.tu_2_others_ids[tu] = others_ids_filtered
            if return_len:
                return len(others_ids_filtered)
            if not self.ACCE_filterB_by_cuda:
                others = np.array(others_filtered)  # P,t_his,J,3
            else:
                others = others_filtered  # P,t_his,J,3
        return others
    def __len__(self):
        return len(list(self.data))


    def __getitem__(self, idx):
        kid, iF, ip = self.data[idx]
        kitchen = self.kid2kitchen[kid]
        objs_raw = kitchen.get_environment(iF)
        objs_pc = [obj_raw.query()
                   for obj_raw in objs_raw]  # [ (n1,3),(n2,3),.... ]

        primary = self.kid__2__ip_2_iF2joints[kid][ip][iF:iF+t_total]  # t_total,J,3
        primary_exists = self.kid__2__ip_2_iF2present[kid][ip][iF:iF+t_total]  # t_total
        primary_exists = torch.tensor(primary_exists) #float
        primary_exists = primary_exists.bool()
        root_primary = primary[t_his-1][ROOT_JOINT_IDX]  # 3
        if 1:
            others = self.get_others(kid, iF, ip)  # P,t_his,J,3
            others = others.float()

        #-------------------------------------
        pose = torch.tensor(   primary   ).float()

        #  objs_pc=[ (n1,3),(n2,3),.... ] -> scene_vert=(n1+n2+...),3
        scene_vert = np.concatenate(objs_pc, axis=0) # N,3
        scene_vert = torch.tensor(scene_vert).float()
        
        scene_origin = torch.clone(pose[t_his-1,ROOT_JOINT_IDX:ROOT_JOINT_IDX+1])#last hist frame root coord as new scene origin
        root_joint = scene_origin
        if 1:
            dist = np.linalg.norm(scene_vert[:, :3] - root_joint, axis=-1)
            if AUG and globals_.TRAIN:
                del root_joint, scene_origin, root_primary
            _max_dist_from_human = self.max_dist_from_human
            idxs = np.where(dist <= _max_dist_from_human)[0]  # filtered points too far from human
            del dist
            #
            len_vidx = len(idxs)
            if CHECK_:
                print_randomly(f"{len_vidx=}", 0.002)
            _num_scene_points = MAX_OBJ_PC_NUMBER
            if len_vidx < _num_scene_points: 
                if len_vidx==0:
                    print(f"{idx=} {kid, iF, ip=}")
                ids = list(range(len_vidx)) + np.random.choice(np.arange(len_vidx),_num_scene_points-len_vidx).tolist()
            else:
                ids = np.random.choice(np.arange(len_vidx), _num_scene_points, replace=False)
            #
            v_idx = idxs[ids]
            scene_vert = scene_vert[v_idx]
            if AUG and globals_.TRAIN:
                scene_vert, pose, others = aug.A(scene_vert, pose, others=others, )
                # re cal
                scene_origin = torch.clone(pose[t_his-1,ROOT_JOINT_IDX:ROOT_JOINT_IDX+1])#last hist frame root coord as new scene origin
                root_joint = scene_origin
            
            objs_pc = scene_vert[:, :3]
            objs_pc = objs_pc.view(1, MAX_OBJ_PC_NUMBER, 3)
            del scene_vert

            objs_semId = torch.empty((others.shape[0],))
        print_randomly(
            f"others: {tuple(others.shape)} objs_pc: {tuple(objs_pc.shape)}", 
            0.001  ,
        )
        objs_pc = objs_pc - scene_origin  
        pose = pose - scene_origin # [90, 21, 3]  T,J,3
        others = others - scene_origin
        item_key =  torch.tensor( [ idx,     ord(kid), iF, ip  ] )  
        if not LOSS_MASK:
            primary_exists= 0

        return pose, objs_pc, scene_origin, objs_semId, item_key, others, primary_exists

