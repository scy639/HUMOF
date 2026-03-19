


from eval_gate import eval_gate,eval_model_interval
import os


if 0:
    from . import auto_select_gpu
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")

num_workers = 10

save_model_interval=10
MMAP_MODE__gta_joint=None
MMAP_MODE__gta_scene="r"
SAVE_csv_dftb:bool = 1


from conf import *
