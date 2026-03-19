if __name__ == '__main__':
    import sys,os; cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(cur_dir, '..')))
    if __package__ is None \
        or __package__ == '': # 4debug
        __package__ = os.path.basename(cur_dir); print(f'{__package__ =}')
import pickle
import numpy as np
import json
from my_py_lib.miscellaneous.DictWithTupleAsKey import DictWithTupleAsKey
from conf import *
from pathlib import Path
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from utils import *
import globals_
from . import aug





ACCE_filterB_by_cuda:bool =1 # whether 1 or 0, will be reset to 0 when __init__ ends

class DatasetHOIm3(Dataset):
    def __init__(self, mode, scene_seen_or_unseen=None, ):
        self.ACCE_filterB_by_cuda = ACCE_filterB_by_cuda
        
        self.step = STEP
        self.max_dist_from_human = 3.5
        self.max_dist_from_primary_to_other = 2.5

        seq_paths = natsorted(HOIM3_preprocess_DIR.glob('*'))
        #filter   folder where only contains  object_names.json
        seq_paths = [seq_path for seq_path in seq_paths if len(list(seq_path.glob('*'))) > 1]
        print(f"{len(seq_paths)=}")
        l_seqName = []
        
        seqName__2__oNames = {}
        seqName__2__iF_2_oR = {}  # oR: [R,R,R,...]
        seqName__2__iF_2_oT = {}
        seqName__2__ip_2_iF2Joints = {}
        TTT3535=1
        if TTT3535:
            def ttt523(ndarr_dtype_object):
                ndarr_dtype_object=ndarr_dtype_object.item()
                ordered_arrays = [ndarr_dtype_object[key] for key in sorted(ndarr_dtype_object.keys())]
                result = np.stack(ordered_arrays, axis=0)
                return result
        for i, seq_path in enumerate(seq_paths):
            seq_name = seq_path.name
            is_testset = (i % 5 == 0)
            
            if (mode == 'train' and not is_testset) or (mode == 'test' and is_testset):
                l_seqName.append(seq_name)
                with open(seq_path / 'object_names.json', 'r') as f:
                    seqName__2__oNames[seq_name] = json.load(f)
                seqName__2__iF_2_oR[seq_name] = np.load(seq_path / 'iFrame_2_objsRs.npy')
                seqName__2__iF_2_oT[seq_name] = np.load(seq_path / 'iFrame_2_objsTs.npy')
                if TTT3535:
                    allow_pickle=1
                else:
                    allow_pickle=0
                ip_2_iF2Joints = np.load(seq_path / 'ip_2_iF2Joints.npy',  allow_pickle=allow_pickle  )
                if TTT3535:
                    ip_2_iF2Joints = ttt523(ip_2_iF2Joints)
                print(f"{ip_2_iF2Joints.shape=}")
                seqName__2__ip_2_iF2Joints[seq_name] = ip_2_iF2Joints
        if 1:
            self.tu_2_others_ids = {}
            path_tu_2_others_ids = DIR_hik__tu_2_others_ids / (
                f'{DATASET_name}-{mode}-{scene_seen_or_unseen}-{STEP}-{t_his}-{t_total}'
                f'-{bool(ENABLE_filterB)}-{THRES_filterB}.json'
                f'-{bool(PRIMARY_FILTER_C)}-{bool(PRIMARY_FILTER_C2)}-{THRES__primary_filterC}.json'
            )  
            if path_tu_2_others_ids.exists():
                print(f'loading {path_tu_2_others_ids}')
                self.tu_2_others_ids = DictWithTupleAsKey.load(path_tu_2_others_ids)
                cache_missed__tu_2_others_ids = False
            else:
                print(f'not found {path_tu_2_others_ids}, will compute it later')
                cache_missed__tu_2_others_ids = True
            # ---------
            self.tu_2_should_filter_primary = {}
            path_tu_2_should_filter_primary = DIR_hik__tu_2_should_filter_primary / (
                f'{DATASET_name}-{mode}-{scene_seen_or_unseen}-{STEP}-{t_his}-{t_total}'
                f'-{bool(PRIMARY_FILTER_C)}-{bool(PRIMARY_FILTER_C2)}-{THRES__primary_filterC}.json'
            ) # python's implicit line joining
            if path_tu_2_should_filter_primary.exists():
                print(f'loading  {path_tu_2_should_filter_primary}')
                self.tu_2_should_filter_primary = DictWithTupleAsKey.load( path_tu_2_should_filter_primary )
                cache_missed__tu_2_should_filter_primary = False
            else:
                print(f'not found  {path_tu_2_should_filter_primary} , will compute it later')
                cache_missed__tu_2_should_filter_primary = True
            # ---------
            if cache_missed__tu_2_others_ids or cache_missed__tu_2_should_filter_primary or PRIMARY_FILTER_C:
                if self.ACCE_filterB_by_cuda:
                    seqName__2__ip_2_iF2Joints = {k:torch.Tensor(v).cuda() for k,v in seqName__2__ip_2_iF2Joints.items()} 
        data = []
        for seqName in tqdm( l_seqName, mininterval=5 ):
            n_p = seqName__2__ip_2_iF2Joints[seqName].shape[0]
            for ip in range(n_p):
                iF2Joints = seqName__2__ip_2_iF2Joints[seqName][ip]
                for iF_start in range(0, len(iF2Joints) - t_total - 1, self.step):
                    tu = (seqName, ip, iF_start)
                    if PRIMARY_FILTER_C:
                        primary = torch.tensor(iF2Joints[iF_start:iF_start+t_total]).float()
                        if tu in self.tu_2_should_filter_primary:
                            assert not cache_missed__tu_2_should_filter_primary
                            should_filter_primary = self.tu_2_should_filter_primary[tu]
                        else:
                            assert cache_missed__tu_2_should_filter_primary
                            should_filter_primary = primary_filterC( primary )
                            self.tu_2_should_filter_primary[tu] = should_filter_primary
                        if should_filter_primary:
                            continue
                    data.append(tu)
        del tu,ip,iF_start,n_p,seqName
        self.data = data
        
        self.seqName__2__ip_2_iF2Joints = seqName__2__ip_2_iF2Joints
        #-----filterB-----
        if ENABLE_filterB:
            data = []
            for idx,tu in enumerate(self.data):
                others_ids = self.get_others(idx )
                len_others = len(others_ids)
                if len_others == 0:
                    continue
                data.append(tu)
            self.data = data
        
        
        print(f'{set([len(_) for _ in self.tu_2_others_ids.values()])=}')
        
        self.l_seqName = l_seqName
        self.seqName__2__oNames = seqName__2__oNames
        self.seqName__2__iF_2_oR = seqName__2__iF_2_oR
        self.seqName__2__iF_2_oT = seqName__2__iF_2_oT
        self.oName__2__oPc = self.load_object_point_clouds()
        
        print(f'{t_total=},{len(self.data)=}')        
        
        self.id2O = [len(self.get_others(i)) for i in range(len(self.data))]
        
        if self.ACCE_filterB_by_cuda:
            self.seqName__2__ip_2_iF2Joints = {k:torch.Tensor(v).cpu().numpy() for k,v in self.seqName__2__ip_2_iF2Joints.items()}
            self.ACCE_filterB_by_cuda=False
        if 1:
            if cache_missed__tu_2_others_ids:
                DictWithTupleAsKey.dump(self.tu_2_others_ids, path_tu_2_others_ids)
                print(f'saved {path_tu_2_others_ids}')
            if cache_missed__tu_2_should_filter_primary:
                DictWithTupleAsKey.dump( self.tu_2_should_filter_primary, path_tu_2_should_filter_primary )
                print(f'saved  {path_tu_2_should_filter_primary}')

    def load_object_point_clouds(self):
        oName__2__oPc = {}
        for path in DIR_downsampled_obj_pc.glob('*.xyz'):
            obj_name = path.stem
            oPc = np.loadtxt(path,skiprows=1)
            oName__2__oPc[obj_name] = torch.tensor(oPc).float()
        return oName__2__oPc

    def __len__(self):
        return len(self.data)

    def get_others(self, idx):
        seqName, ip, iF_start = self.data[idx]
        tu = (seqName, ip, iF_start)

        # Check if result is already cached
        if tu in self.tu_2_others_ids:
            others_ids = self.tu_2_others_ids[tu]
            return others_ids

        n_p = self.seqName__2__ip_2_iF2Joints[seqName].shape[0]
        others_ids = [i for i in range(n_p) if i != ip]
        
        primary = self.seqName__2__ip_2_iF2Joints[seqName][ip][iF_start:iF_start+t_total]
        root_primary = primary[t_his-1][ROOT_JOINT_IDX]

        others_filtered = []
        others_ids_filtered = []
        for other_id in others_ids:
            other = self.seqName__2__ip_2_iF2Joints[seqName][other_id][iF_start:iF_start+t_total]
            if not ENABLE_filterB:
                root_other = other[t_his-1][ROOT_JOINT_IDX]
                dis_to_primary = np.linalg.norm(root_other-root_primary)
                if dis_to_primary > self.max_dist_from_primary_to_other:
                    continue
            else:
                if filterB(other,primary,self.ACCE_filterB_by_cuda):
                    continue
            others_filtered.append(other[:t_his])
            others_ids_filtered.append(other_id)

        # Cache the results
        self.tu_2_others_ids[tu] = others_ids_filtered

        return others_ids_filtered

    def __getitem__(self, idx):
        seqName, ip, iF_start = self.data[idx]
        iF_lastObserved = iF_start + t_his - 1
        iF_end = iF_start + t_total 
        
        oNames = self.seqName__2__oNames[seqName]
        iF_2_oR = self.seqName__2__iF_2_oR[seqName]
        iF_2_oT = self.seqName__2__iF_2_oT[seqName]
        ip_2_iF2Joints = self.seqName__2__ip_2_iF2Joints[seqName]
        
        objs_pc = []
        for oName in oNames:
            oPc = self.oName__2__oPc[oName]
            oR = iF_2_oR[iF_lastObserved][oNames.index(oName)]
            oT = iF_2_oT[iF_lastObserved][oNames.index(oName)]
            
            oPc = oPc @ oR + oT
            # tensor to fp32
            oPc = oPc.float()
            
            objs_pc.append(oPc)
        
        primary = ip_2_iF2Joints[ip][iF_start:iF_end]
        others_ids = self.get_others(idx)
        others = ip_2_iF2Joints[others_ids, iF_start:iF_start+t_his]
        
        root_primary = primary[t_his-1][ROOT_JOINT_IDX]
        others = torch.tensor(others).float()

        pose = torch.tensor(primary).float()

        #  objs_pc=[ (n1,3),(n2,3),.... ] -> scene_vert=(n1+n2+...),3
        scene_vert = torch.cat(objs_pc, dim=0)
        
        scene_origin = pose[t_his-1, ROOT_JOINT_IDX:ROOT_JOINT_IDX+1].clone()
        root_joint = scene_origin
        if 1:
            dist = np.linalg.norm(scene_vert[:,:3] - root_joint, axis=-1)
            if AUG and globals_.TRAIN:
                del root_joint,scene_origin,root_primary
            _max_dist_from_human = self.max_dist_from_human
            idxs = np.where(dist <= _max_dist_from_human )[0] # filtered points too far from human
            del dist
            #
            len_vidx = len(idxs)
            if CHECK_:
                print_randomly(f"{len_vidx=}", 0.002)
            _num_scene_points = MAX_OBJ_PC_NUMBER
            if len_vidx < _num_scene_points: 
                if len_vidx==0:
                    print(f"{idx=} {ip=}")
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
        item_key =  torch.tensor( [ idx,     ip, iF_start,  ] )  
        if not LOSS_MASK:
            primary_exists= 0

        return pose, objs_pc, scene_origin, objs_semId, item_key, others, primary_exists
