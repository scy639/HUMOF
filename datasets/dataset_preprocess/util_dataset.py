import torch
from conf import *
from utils import *


def filterB(
    other:np.ndarray,
    primary:np.ndarray,
    thres_filterB,
    ACCE_filterB_by_cuda:bool=False,
)->bool: #True: filter it; False: keep it
    """
    other: t_toal,J,3
    primary: t_toal,J,3

    keep when:
    1. the min dis bet other <--> primary  < THRE in IF_start~IF_end
        min means cal j2j mat (shape: J*J) and get the min,
    """
    IF_start = t_his-t_his//3
    IF_end = t_total
    mat = other[ IF_start:IF_end, :, None, :] - primary[ IF_start:IF_end, None, :, :] # T,J,J,3
    assert mat.shape[1:]==(NUM_J,NUM_J,3)

    if not ACCE_filterB_by_cuda:
        # np  version
        dis_mat = np.linalg.norm( mat, axis=-1)  # T,J,J
        assert dis_mat.shape[1:]==(NUM_J,NUM_J)
        min_distances = np.min(dis_mat, axis=(1, 2))  # T
        if   np.any(min_distances < thres_filterB):
            return False
        else:
            return True
    else:
        # cuda version:
        dis_mat = torch.norm( mat, dim=-1)  # T,J,J
        min_distances = dis_mat.min(1).values  #   T x J
        min_distances = min_distances.min(1).values  #   T
        if   torch.any(min_distances < thres_filterB):
            return False
        else:
            return True

def primary_filterC( primary: torch.Tensor, thres__primary_filterC ) -> bool: #True: filter it; False: keep it
    """
    primary: t_toal,J,3
    keep when:
    1. root(T,3)'s movement range > THRE in IF_start~IF_end
    """
    assert primary.shape==(t_total,NUM_J,3)
    IF_start = t_his-t_his//3
    IF_end = t_total
    if PRIMARY_FILTER_C2:
        for index_joint in range(  NUM_J ):
            a = primary[ IF_start:IF_end, index_joint ] # T,3
            assert a.shape[1:]==(3,)
            max_distance, point_pair, index_of_point_pair = find_furthest_point_pair( a )
            max_distance = max_distance.item()
            if max_distance < thres__primary_filterC:
                continue
            else:
                return False
        return True
    else:
        _primary_root = primary[ IF_start:IF_end, ROOT_JOINT_IDX ]
        assert _primary_root.shape[1:]==(3,)
        max_distance, point_pair, index_of_point_pair = find_furthest_point_pair( _primary_root )
        # print(f"{max_distance=}")
        max_distance = max_distance.item()
        if max_distance < thres__primary_filterC:
            return True
        else:
            return False
