from conf import *
import torch

device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
if 1:
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    