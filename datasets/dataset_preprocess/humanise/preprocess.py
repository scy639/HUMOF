import os, sys
import argparse
import random
import glob
import shutil
import smplx
import torch
import pickle
from tqdm import tqdm
import trimesh
import pandas as pd
import numpy as np
from natsort import natsorted
random.seed(0)

from pathlib import Path
OUTPUT=Path("../../../data/humanise/processed_")
OUTPUT.parent.parent.parent.mkdir(parents=0,exist_ok=1)
OUTPUT.parent.parent.mkdir(parents=0,exist_ok=1)
OUTPUT.parent.mkdir(parents=0,exist_ok=1)
OUTPUT.mkdir(parents=0,exist_ok=1)


parser = argparse.ArgumentParser()
parser.add_argument('--humanise_dir', type=str)
parser.add_argument('--scene_dir', type=str)
parser.add_argument('--smplx_dir', type=str)
args = parser.parse_args()

body_model_neutral = smplx.create(
        args.smplx_dir, model_type='smplx',
        gender='neutral', ext='npz',
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=1,
).to(device='cpu')
@torch.no_grad()
def smplx_params_to_meshes(transl, orient, betas, body_pose, hand_pose):
    verts = []
    joints=[]
    for i in range(len(transl)):
        output = body_model_neutral(
            return_verts=True,
            transl=transl[i:i+1],
            global_orient=orient[i:i+1],
            betas=betas,
            body_pose=body_pose[i:i+1],
            left_hand_pose=hand_pose[i:i+1, :45],
            right_hand_pose=hand_pose[i:i+1, 45:]
        )
        vertices = output.vertices.detach().cpu().numpy()
        verts.append(vertices)
        
        
        j=output.joints
        assert j.shape[0]==1
        assert j.shape[2]==3
        j=j[0,:,:]
        j_humanise=j
        j_gta=j_humanise[:,:]
        j_gta=j_gta.numpy()
        joints.append(j_gta)
        
        

    verts = np.concatenate(verts, axis=0)
    faces = body_model_neutral.faces

    return verts, faces,joints

def load_humanise_motions(motion_path, annotations, scene_dir):
    index = int(motion_path.split('/')[-1].split('.')[0])
    scene_id = annotations.loc[index]['scene_id']
    path_s=OUTPUT/f'{scene_id}.xyz'
    poses, betas = pickle.load(open(motion_path, 'rb'))

    trans = torch.from_numpy(poses[:, :3]).float()
    root_orient = torch.from_numpy(poses[:, 3:6]).float()
    pose_body = torch.from_numpy(poses[:, 6:69]).float()
    pose_hand = torch.from_numpy(poses[:, 69:]).float()
    betas = torch.from_numpy(betas).float().unsqueeze(0)



    verts, faces ,joints_gta= smplx_params_to_meshes(trans, root_orient, betas, pose_body, pose_hand)
    if 1:
        #save joints_gta as npy
        joints_gta=np.array(joints_gta)
        np.save(OUTPUT/f'{index}.npy', joints_gta)
        print('save joints_gta shape:',joints_gta.shape)
    if 1:
        if path_s.exists():
            print(f'path_s exists.')
            return
    
    

    texts = [annotations.loc[index]['text']]

    scene_trans = np.array([
        float(annotations.loc[index]['scene_trans_x']),
        float(annotations.loc[index]['scene_trans_y']),
        float(annotations.loc[index]['scene_trans_z']),
    ], dtype=np.float32)
    scene = trimesh.load(os.path.join(scene_dir, f'{scene_id}/{scene_id}_vh_clean_2.ply'), process=False)
    scene.apply_translation(scene_trans)
    
    
    
    
    v=scene.vertices
    assert v.shape[1]==3,v.shape
    if 1:
        with open(path_s, 'w') as f:
            for x, y, z in v:
                f.write(f'{x} {y} {z}\n')
        print('save xyz shape:',v.shape)
    
    
    

    return verts, faces, texts, scene


shutil.copy( Path(args.humanise_dir) / "annotation.csv", OUTPUT)

motion_paths = natsorted(glob.glob(os.path.join(args.humanise_dir, 'motions/*.pkl')))
anno_file = pd.read_csv(os.path.join(args.humanise_dir, 'annotation.csv'))
total_amount = anno_file.shape[0]
assert total_amount == len(motion_paths)
# print(f"==>> motion_paths[:20]: {motion_paths[:20]}")

if  1  :# multi processor
    def process_motion(motion_path):
        return load_humanise_motions(motion_path, anno_file, args.scene_dir)
    import multiprocessing as mp
    p = mp.Pool(processes=   min(  mp.cpu_count()  ,  8   )       )
    print(f"==>> mp.cpu_count(): {mp.cpu_count()}")
    p.map(  process_motion  , motion_paths)
    p.close()
    p.join()
else:
    for motion_path in tqdm(motion_paths):    
        _ = load_humanise_motions(motion_path, anno_file, args.scene_dir)
