"""
should be read-only
"""
from pathlib import Path
import os,sys



# _gta_dir  =  Path('/data/shared/suncy/GTA-IM_Dataset--processed')
_gta_dir  =  Path('./data/GTA-IM_Dataset/processed')
gta_data_path  =  _gta_dir  /  'data_v2_downsample0.02-suncy2.2smallA'
gta_idx_path  =  _gta_dir  /  'processed_seq_pkl-perFrame-suncy2.2'



DIR_cache_4_Dataset = Path( "./data/cache_4_Dataset" )

#------------above conf rarely changed------------------
from conf0 import *

"""
for humanise
"""
HUMANISE_DIR= "./data/humanise"
HUMANISE_DIR=Path(HUMANISE_DIR)
J_S_DIR=HUMANISE_DIR/'processed_'
S_DIR=J_S_DIR


HIK_raw_DIR= "./data/hik/data"
HIK_preprocess_DIR= "./data/hik_preprocessB/H25F50"
HIK_raw_DIR=Path(HIK_raw_DIR)
HIK_preprocess_DIR=Path(HIK_preprocess_DIR)
DIR_hik__tu_2_others_ids = HIK_preprocess_DIR / 'tu_2_others_ids'
DIR_hik__tu_2_should_filter_primary = HIK_preprocess_DIR / 'tu_2_should_filter_primary'


HOIM3_preprocess_DIR = "./data/HOIm3_preprocess/A-FRAME_DSratio=2"
DIR_downsampled_obj_pc='./data/HOIm3_preprocess/_oName__2__oPc/MAX_N=1000'
HOIM3_preprocess_DIR=Path( HOIM3_preprocess_DIR )
DIR_downsampled_obj_pc=Path( DIR_downsampled_obj_pc )
#-----------------------------------------------------



SPLIT_FILE_NAME='C' # B: same motion; C: from MutualDistance author




MAX_OBJ_PC_NUMBER=1000
# OBJ_PC_DOWNSAMPLE:str='FPS'
OBJ_PC_DOWNSAMPLE:str='RANDOM'  # also adopted by ContAware
SAP_as_objNode:bool =True   # SAP: set abstracted point
GCN_encode_ego:bool = True
if DATASET_name=='gta':
    GCN_encode_ego = False # as 1019-gcnFix1_resB* worse than 1019
GCN__num_stage = 5
if 1:
    DS_SA :bool = True # SE after ds in SA (in res branch
    DS_CA :bool = True
    DS_FFN:bool = True
    DS_ctx:bool = False
    if DS_ctx:
        DS_ctx_beforeLN:bool = True
        DS_ctx_afterLN :bool = False
        assert not (DS_ctx_beforeLN and DS_ctx_afterLN)
if 1:
    TR_ctx :bool = False # token reweight (TR) for ctx token

if DATASET_name == 'hik':
    from datasets.dataset_preprocess.hik import *
    t_his=25
    t_pred=50
    MULTI_PERSON_MODE = True
elif DATASET_name == 'hoi':
    from datasets.dataset_preprocess.hoim3 import *
    t_his=30
    t_pred=60
    MULTI_PERSON_MODE = True
elif DATASET_name == 'gta':
    from datasets.dataset_preprocess.gtaim import *
    t_his=30
    t_pred=60
    MULTI_PERSON_MODE = False
elif DATASET_name == 'humanise':
    from datasets.dataset_preprocess.humanise.confs import *
    t_his=15
    t_pred=30
    MULTI_PERSON_MODE = False
else:
    raise ValueError(f"{DATASET_name=}")

class PC_FILTER:
    N1=50
    N2=100
    R2=0.1
    R=0.6
    MIN_OBJ_NUM=5


dct_n= 20
sigma= 0.02
sigma_4pe= 0.2
NORM_OBJ = True
CONCAT_OBJ_PE = True
INTRODUCE_semantic_label:bool = 0
if MULTI_PERSON_MODE:
    MIN_OTHER_NUM=0
    HHI_B:bool = 1 # HHI: the modeing of  Human to Human Interaction. type B
    ENCODE_others_locally:bool = 1
    OL_SHARE_primary_SA:bool = 0 # ENCODE_others_locally's sub-config
    GCN_oth:bool = 1
    OL_cat_perJtoken_in_last_layers:bool = 1 # ENCODE_others_locally's sub-config
    NO_perP_token_in_last_layers:bool = 1 # last layers use perJ token only, no perP token. OL_cat_perJtoken_in_last_layers sub-config (use together)
    


if DATASET_name=='hik':
    LOSS_L1 = True   # for hik
else:
    LOSS_L1 :bool = False
LOSS_MASK :bool =   0
DEEP_SUPERVISION:bool = False
AUG :bool= 1
if MULTI_PERSON_MODE:
    BS=1
else:
    BS=24
lr=5.e-4
if DATASET_name == 'humanise':
    DROPOUT=0.2
else:
    DROPOUT=0.1
class DROPOUT_ctx_token:
    enable:bool = 0
    prob:float = 0.1 # dropout rate
if MULTI_PERSON_MODE:
    ADAM_eps = 1e-6
else:
    ADAM_eps = 1e-8 # default value in Adam
num_epoch_fix=1
if MULTI_PERSON_MODE:
    num_epoch=  70
else:
    num_epoch= 50
# ds: Dct reScaling
APPLY_ds_since:int = 0 # if 0, ds throughout (old default)
LEARN_ds_since:int = num_epoch//2 # if 0, learn throughout (old default)
assert APPLY_ds_since<=LEARN_ds_since, 'apply ds epoch range must cover learn ds'



Preprocess_in_dataloader:bool  =  0

result_id = f"{DATASET_name}-releaseV0.1"
# result_id= 'tttt34546'
print(f"----------------result_id:  {result_id}  -------------------")
result_dir=os.path.join(   './results'  , result_id     )

#------non-config------------------
t_total=t_his+t_pred


if MULTI_PERSON_MODE:
    if DATASET_name == 'hik':
        NUM_J=29
        ROOT_JOINT_IDX = 0
    elif DATASET_name == 'hoi':
        NUM_J=24
        ROOT_JOINT_IDX = 0  
    else:
        raise ValueError(f"{DATASET_name=}")
else:
    if DATASET_name == 'gta':
        NUM_J=21
        ROOT_JOINT_IDX = 14
    elif DATASET_name == 'humanise':
        NUM_J=127
        ROOT_JOINT_IDX = 0
root_joint_idx=ROOT_JOINT_IDX

#
if not os.path.exists("results"):     os.mkdir("results")
if not os.path.exists(result_dir):     os.mkdir(result_dir)
#
model_dir =Path('checkpoints')
model_dir.mkdir(exist_ok=True)
model_dir = model_dir / result_id
model_dir.mkdir(exist_ok=True)
model_path = os.path.join(  model_dir,   '%d.pth'  )

if DATASET_name=='hik':
    DIR_hik__tu_2_others_ids.mkdir(exist_ok=True)
    DIR_hik__tu_2_should_filter_primary.mkdir(exist_ok=True)