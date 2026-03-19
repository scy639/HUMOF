from conf import *
import torch


l__dct_scale_4_dim2=[
    torch.empty(  ( 3*dct_n ,),  dtype=torch.float32).cuda() 
    for _ in range(6)
]
l__i_dct_2_gradScale=[
    [1,]*8  + [0.7]*7 + [0.3]*5 , # layer 0
    [1,]*10 + [0.7]*6 + [0.3]*4 , # layer 1
    [1,]*12 + [0.7]*5 + [0.5]*3 , # layer 2
    [1,]*14 + [0.8]*4 + [0.6]*2 , # layer 3
    [1,]*20,  # layer 4
    [1,]*20,  # layer 5
]
assert len(l__i_dct_2_gradScale)==6
for i_layer in range(6):
    i_dct_2_gradScale = l__i_dct_2_gradScale[i_layer]
    assert len(i_dct_2_gradScale)==dct_n, i_dct_2_gradScale
    for i_xyz in range(3):  # i_xyz=0 means now you are setting x
        shift = i_xyz*dct_n
        for i_dct,gradScale in enumerate( i_dct_2_gradScale ):
            l__dct_scale_4_dim2[i_layer] [shift  +  i_dct] = gradScale
def grad_hook(grad, dct_scale_4_dim2):
    # if CHECK_:
    #     assert grad.shape==(BS,21,3*dct_n), grad.shape
    # grad[:,:,dct_scale_4_dim2] = 0
    grad = grad * dct_scale_4_dim2[None,None,:]
    return grad
grad_hooks=[
    lambda grad: grad_hook(grad,dct_scale_4_dim2=l__dct_scale_4_dim2[0]  ),
    lambda grad: grad_hook(grad,dct_scale_4_dim2=l__dct_scale_4_dim2[1]  ),
    lambda grad: grad_hook(grad,dct_scale_4_dim2=l__dct_scale_4_dim2[2]  ),
    lambda grad: grad_hook(grad,dct_scale_4_dim2=l__dct_scale_4_dim2[3]  ),
    lambda grad: grad_hook(grad,dct_scale_4_dim2=l__dct_scale_4_dim2[4]  ),
    lambda grad: grad_hook(grad,dct_scale_4_dim2=l__dct_scale_4_dim2[5]  ),
]