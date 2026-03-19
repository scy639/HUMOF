from conf import *
import numpy as np
import torch
import os
import random
from conf_torch import *
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        return point
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point



def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m


class DCT_IDCT:
    dct_m, idct_m = get_dct_matrix(t_total, is_torch=True)
    dct_m = dct_m.to(dtype=dtype, device=device)[:dct_n]
    idct_m = idct_m.to(dtype=dtype, device=device)[:,:dct_n]
    assert tuple(dct_m.shape)==(dct_n,t_total),dct_m.shape
    assert tuple(idct_m.shape)==(t_total,dct_n),dct_m.shape
    @classmethod
    def perform_DCT(  cls,x,    ):
        dct_m=cls.dct_m
        """
        dct_m: dct_n,T
            
        x: ..., T,n
        ret:  ..., dct_n,n
        """
        x1=x
        orig_shape_=x1.shape[:-2]
        assert dct_m.shape[1]==x1.shape[-2]
        assert dct_m.shape[0]==dct_n
        # # ..., T,3 -> -1,T,3
        # x1= x1.view(-1, *x1.shape[-2:])
        dct=torch.matmul(input=dct_m,other=x1)
        # # -1,T,3 -> original ... , 
        # dct=dct.view(   *orig_shape_, ...  )
        """
        assert dct.shape[:-2]==orig_shape_
        assert dct.shape[-2]==dct_n
        assert dct.shape[-1]==x1.shape[-1]
        """
        assert dct.shape==(  *orig_shape_, dct_n, x1.shape[-1]  )
        return dct
    @classmethod
    def perform_IDCT( cls, x,   ):
        idct_m=cls.idct_m
        """
        idct_m: T,T
            
        x: ..., dct_n,n
        ret:  ..., T,n
        """
        x1=x
        orig_shape_=x1.shape[:-2]
        assert x1.shape[-2]==idct_m.shape[1]==dct_n
        T=idct_m.shape[0]
        y=torch.matmul(input=idct_m,other=x1)
        assert y.shape==(  *orig_shape_, T, x1.shape[-1]  )
        return y


from torch.optim import lr_scheduler
def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler

def print_randomly(a ,p=1):
    """
    p is probability
    """
    if p<1:
        if random.random()>=p:
            return
    print(a)


def get_latest_ckpt_A(dir_,fmt):
    """
    get the latest ckpt,
    
    fmt:   '%d.pth'
    """
    suffix=fmt.split('.')[-1] # 'pth'
    names=os.listdir(dir_)
    l_int=[int(name.split('.')[0]) for name in names if name.endswith( '.'+suffix )]
    l_int.sort()
    if len(l_int)==0:
        return None
    return os.path.join(dir_,fmt%l_int[-1]), l_int[-1]
def get_latest_ckpt_B(fullpath_fmt=None):
    """
    fullpath_fmt:     '/FULL_PATH/%d.pth' 
    """
    if fullpath_fmt is None:
        fullpath_fmt=model_path
    suffix=fullpath_fmt.split('.')[-1] # 'pth'
    names=os.listdir(os.path.dirname(fullpath_fmt))
    l_int=[int(name.split('.')[0]) for name in names if name.endswith( '.'+suffix )]
    l_int.sort()
    print(  f"{fullpath_fmt=} {names=}"  )
    if len(l_int)==0:
        raise Exception
    return fullpath_fmt%l_int[-1], l_int[-1]
    
def find_furthest_point_pair(points   ):
    """
    points: Tensor   N,3
    """
    import torch
    def unravel_index(index, shape):
        # calc row and col indices
        if 0: # get   UserWarning: __floordiv__ is deprecated
            row = index // shape[1]
        else:# use torch.div with floor rounding
            row = torch.div(index, shape[1], rounding_mode='floor')
        col = index % shape[1]
        return (row, col)

    # expand dims for broadcasting
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, 1, 3) - (1, N, 3)
    
    # calc squared Euclidean distance
    dist_matrix = torch.sum(diff ** 2, dim=-1)  # (N, N)
    
    # find max distance index
    max_idx = torch.argmax(dist_matrix)
    max_idx = unravel_index(max_idx, dist_matrix.shape)
    
    # get furthest point pair and max distance
    max_distance = torch.sqrt(dist_matrix[max_idx])
    point_pair = (points[max_idx[0]], points[max_idx[1]])
    index_of_point_pair = ( max_idx[0],max_idx[1] )
    
    return max_distance, point_pair, index_of_point_pair
    