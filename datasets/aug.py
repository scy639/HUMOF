import math
import torch
import torchgeometry
def A(
    scene_vert, # N,4
    joints, # T,J,3
    others=None, # P,T,J,3
):
    orig_scene_shape=scene_vert.shape
    orig_joints_shape=joints.shape
    
    if orig_scene_shape[1]>3:
        orig_scene_vert=scene_vert
        scene_vert=scene_vert[:,:3]# N,3
    
    dtype=scene_vert.dtype
    ang = torch.zeros([ 3], dtype=dtype)
    ang[2] = torch.rand( (), dtype=dtype)*math.pi # 3
    rand_rot = torchgeometry.angle_axis_to_rotation_matrix(ang[None,:]) # 0,4,4
    rand_rot=rand_rot[0] # 4,4
    rand_tran = torch.rand([3],dtype=dtype)-0.5 # 3
    rand_rot[:3,3] = rand_tran # 4,4
    scene_vert = torch.matmul(rand_rot[:3,:3],scene_vert.transpose(0,1)).transpose(0,1) \
                    + rand_rot[:3,3].unsqueeze( 0 ) # N,3 = N,3 + 0,3 (or rand_rot[None,:3,3] )
    joints = torch.matmul(rand_rot[:3,:3],joints.transpose(1,2)).transpose(1,2) \
                    + rand_rot[None,None,:3,3] # T,J,3 = T,J,3 + 0,0,3
    if others is not None:
        for i_p,other in enumerate(others):
            others[i_p] = torch.matmul(rand_rot[:3,:3],other.transpose(1,2)).transpose(1,2) \
                            + rand_rot[None,None,:3,3]
    
    if orig_scene_shape[1]>3:
        orig_scene_vert[:,:3]=scene_vert
        scene_vert=orig_scene_vert
    
    assert scene_vert.shape==orig_scene_shape
    assert joints.shape==orig_joints_shape
    if others is not None:
        return scene_vert,joints,others
    return scene_vert,joints