"""
"""
from conf import *
from conf2 import *
sys.path.append(os.getcwd())
# ccc-------------------------------
cp_iter = 0    # iter number of ckpt to be loaded. if 0, init model. can be 'auto'
model_path_trained = model_path
TRAIN=   bool(    1    )
SHUFFLE_when_train = True
SHUFFLE_when_test  = False
prefetch_factor=       2       if num_workers>0 else 2

max_batch_objs=60
max_batch_size=48
if MULTI_PERSON_MODE:
    max_batch_objs=256
    max_batch_size=48

scene_seen_or_unseen= None  # None | str  valid only when test
if MULTI_PERSON_MODE:
    if DATASET_name == 'hik':
        from datasets.dataset_hik import *
        dataset_cls= DatasetHik
    elif DATASET_name == 'hoi':
        from datasets.dataset_hoim3 import *
        dataset_cls= DatasetHOIm3
else:
    if DATASET_name == 'humanise':
        from datasets.dataset_humanise import *
        dataset_cls= DatasetHumanise
    elif DATASET_name == 'gta':
        from datasets.dataset_gtaim import *
        dataset_cls= DatasetGTA

weight_decay = 1e-6
#---------------------------------


import globals_
import math

import time
import copy
from torch import Tensor, optim
from torch.cuda.amp import autocast, GradScaler
from models import pipelines
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from utils import *
from socket import gethostname

train_loss_joints:int=None
train_loss_root:int=None
TMP1=0
ct=0
TMP2=1
def loss_function(joints, y, 
    primary_exists=None, # B,t_toal
                #   root_traj, root_joint_idx,
                #    epoch,  **kwargs,
                   ):
    global ct
    """
    joints: [bs,t_pred,jn,3] 
    y: [bs,t_pred,jn,dim]
    """
    global train_loss_joints,train_loss_root

    # lossA1:
    # loss_all =  torch.mean((joints  - y ).pow(2).sum(dim=-1)) # may cause pose train better than path
    # lossA2:
    bs, _, jn, _ = y.shape
    jidx = np.setdiff1d(np.arange(jn),[root_joint_idx])
    root_traj = y[:,:,root_joint_idx]
    loss_joints = joints[:,:,jidx] - y[:,:,jidx]
    loss_root = joints[:,:,root_joint_idx]-root_traj
    if TMP1:
        print(f'{ct=}')
        if ct==9:
            print(f'{ct=}')
    loss_joints = loss_joints.pow(2).sum(dim=-1)
    loss_root = loss_root.pow(2).sum(dim=-1)
    if primary_exists is not None:
        assert primary_exists.shape==(bs,t_total), primary_exists.shape
        primary_exists = primary_exists[:,t_his:]
        if TMP1:
            all_true = primary_exists.all()
            if not all_true:
                false_positions = torch.nonzero(~primary_exists, as_tuple=False)
                false_positions=false_positions.detach().cpu().numpy()  
                print(  false_positions   )
                print(f'{1=}')
        assert primary_exists.shape==loss_joints.shape[:2], (primary_exists.shape,loss_joints.shape)
        assert primary_exists.shape==loss_root.shape, (primary_exists.shape,loss_root.shape)
        loss_joints = loss_joints * primary_exists[:,:,None]
        loss_root = loss_root * primary_exists
    if LOSS_L1:# L1 dis instead of L2 (ie, Euclidean). L2 is x^2+y^2+z^2, no sqrt
        loss_joints = loss_joints.sqrt()
        loss_root = loss_root.sqrt()
    loss_joints = loss_joints.mean()
    loss_root = loss_root.mean()
    loss_all = loss_joints + loss_root
    if TMP1:
        print(   loss_joints.detach().cpu().numpy().item()     )
        print(   loss_root.detach().cpu().numpy().item()     )
        print(   loss_all.detach().cpu().numpy().item()     )
        if TMP2:
            loss_item=loss_all.detach().cpu().numpy().item()
            if loss_item>999:
                print(f'{loss_item=}')
                loss_all=0
        is_nan = torch.isnan(loss_all)
        if is_nan:
            print(f'{1=}')
        else:
            print(f'not nan')
        ct+=1
    if torch.isnan(loss_all):
        print('loss nan')

    train_loss_joints+=loss_joints.detach()
    train_loss_root+=loss_root.detach()
    return loss_all ,{
        'loss_joints':loss_joints,
        'loss_root':loss_root,
    }


def print_obj_num_in_a_batch(objs_pc):
    l_num = [objs_pc.shape[0] for objs_pc in objs_pc]
    total = sum(l_num)
    print(f"print_obj_num_in_a_batch: {l_num=}  {total=}")


@torch.no_grad()
def train(epoch,TRAIN:bool,dataset ):
    global train_loss_joints,train_loss_root
    globals_.TRAIN=TRAIN
    globals_.cur_epoch=epoch
    if TRAIN:
        model.train()
        SHUFFLE = SHUFFLE_when_train
    else:
        model.eval()
        SHUFFLE = SHUFFLE_when_test
    t_s=time.time()
    root_idx = [root_joint_idx*3,root_joint_idx*3+1,root_joint_idx*3+2]
    dftb.set_or_new_k( epoch )

    print(f"==>> len(dataset): {len(dataset)}")
    if MULTI_PERSON_MODE:
        DYN_BS=  1     # dynBS
    else:
        DYN_BS=  0
    if  DYN_BS  :
        from datasets.DynamicBatchSampler import DynamicBatchSampler
        batch_sampler = DynamicBatchSampler.wrap_A(
            dataset=dataset,
            #
            max_batch_tokens=max_batch_objs, 
            max_batch_size=max_batch_size,
            #
            SHUFFLE=SHUFFLE,
            seed=seed,
        )
        collate_fn= lambda batch:pipelines.Preprocess.truncateA( batch)
        #
        bs=1  #batch_sampler option is mutually exclusive with batch_size
        shuffle=None
        drop_last=None
    else:
        batch_sampler=None
        collate_fn=( lambda batch:pipelines.Preprocess.B( batch)  )  if Preprocess_in_dataloader else None
        #
        bs=BS
        shuffle=SHUFFLE
        drop_last=True
        
    generator = DataLoader(
        dataset,batch_size=bs,
        collate_fn=collate_fn, 
        batch_sampler=batch_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        #    pin_memory=True,
        drop_last = drop_last,
        prefetch_factor=prefetch_factor,
    )
    if DYN_BS:   
        batch_sampler.set_epoch(epoch)#  must be set each ep
        if 1:
            # print(   dataset.id2O  )
            print(  f"{batch_sampler._batches[-3:]=}"   )
            _batches_2= [  [ dataset.id2O[id_] for id_ in batch] for batch in  batch_sampler._batches  ]
            print(f"{_batches_2[-10:]=}")

    pose_err = np.zeros(t_pred)
    path_err = np.zeros(t_pred)
    all_err = np.zeros(t_pred)
    
    train_losses=0
    train_loss_joints=0
    train_loss_root=0
    total_num_sample = 1e-20

    # y_for_save = {}
    for _tmp in tqdm(generator,mininterval=20):
    # for _tmp in generator:
        if  Preprocess_in_dataloader :
            assert not MULTI_PERSON_MODE
            pose, objs_pc, scene_origin, objs_semId, item_key,\
            joints,\
            joints_repeat_O, dcts, objs_pc_bsFlat, is_cont_dct = _tmp
        else:
            if MULTI_PERSON_MODE:
                (
                    pose, 
                    objs_pc, 
                    scene_origin, 
                    objs_semId, 
                    item_key,
                    others,
                    primary_exists,
                ) = _tmp
                if CHECK_:
                    assert pose.shape[1]==t_total
                primary = pose[:, :t_his].clone()
                primary = primary.to(device=device)
                (
                    joints,
                    joints_repeat_O, 
                    dcts,
                    objs_pc_bsFlat,
                    objs_semId,
                    is_cont_dct,
                    others,
                )=pipelines.Preprocess.A_multiPerson(_tmp[:-1])
                if CHECK_:
                    assert joints.shape[1]==t_total
                    assert others.shape[2]==t_his
                others = others[:, :, :t_his]
                others = others.to(device=device)
            else:
                pose, objs_pc, scene_origin, objs_semId, item_key= _tmp
                joints,\
                joints_repeat_O, dcts, objs_pc_bsFlat, objs_semId, is_cont_dct = pipelines.Preprocess.A(
                    pose, objs_pc, scene_origin, objs_semId, item_key,
                )
                primary=None
                others=None
        if 1:
            bs=pose.shape[0]
            num_obj = objs_pc.shape[1]
            nj = pose.shape[2]
            BO = bs*num_obj
        if  0:
            print_obj_num_in_a_batch(objs_pc)
        if CHECK_:
            if  torch.isnan(joints_repeat_O).any(): print(f'{item_key=}')
            if  torch.isnan(dcts).any():  print(f'{item_key=}')
            if  torch.isnan(objs_pc_bsFlat).any(): print(f'{item_key=}')
            if  torch.isnan(is_cont_dct).any(): print(f'{item_key=}')
            if primary is not None  and  torch.isnan(primary).any(): print(f'{item_key=}')
            if others  is not None  and  torch.isnan(others ).any(): print(f'{item_key=}')
        with torch.set_grad_enabled(TRAIN):
            with autocast(enabled=True):
                dcts_pred , q_from_inter_layers = model(
                    x0=joints_repeat_O[:, :t_his].reshape([BO, t_his, -1]).transpose(0, 1), 
                    dcts=dcts,# B,J*3,DCT
                    scene=objs_pc_bsFlat.transpose(1, 2),
                    cont_dct=is_cont_dct,
                    objs_semId=objs_semId,
                    #
                    primary=primary,
                    others=others,
                )  # B,J*3,DCT
                y= DCT_IDCT.perform_IDCT(  x=dcts_pred.permute(0,2,1)  )# B,T,J*3
                # -> B,T,J,3
                y=y.view(  (bs, t_total, nj, 3)  )# 14 is traj
                y=y[:,t_his:,...]
                if LOSS_MASK :
                    assert primary_exists is not None
                    primary_exists=primary_exists.to(device)
                else:
                    primary_exists = None
                loss,loss_dic=loss_function(joints[:, t_his:]   , y, 
                    primary_exists=primary_exists,
                                )
                if torch.isnan(loss):
                    print('loss-nan')
                    _item_key=item_key.numpy()
                    print(f"==>> _item_key: {_item_key}")
                del primary_exists

        if   TRAIN:
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # # grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1000)
            # grad_norm = 0
            # train_grad += grad_norm
            train_losses += loss.detach()
            total_num_sample += 1
            # break
        else:
            joints = joints[:, t_his:]
            assert joints.shape==y.shape

            """mpjpe error"""
            path_err += (y[:, :, ROOT_JOINT_IDX] - joints[:, :, ROOT_JOINT_IDX]).norm(dim=-1).sum(dim=0).cpu().data.numpy()
            pose_idx = np.setdiff1d(np.arange(NUM_J), ROOT_JOINT_IDX)
            pose_err += (y[:, :, pose_idx] - joints[:, :, pose_idx]).norm(dim=-1).mean(dim=-1).sum(dim=0).cpu().data.numpy()

            y[:, :, pose_idx] = y[:, :, pose_idx] + y[:, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
            joints[:, :, pose_idx] = joints[:, :, pose_idx] + joints[:, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
            all_err += (y - joints).norm(dim=-1).mean(dim=-1).sum(dim=0).cpu().data.numpy()

            # if args.save_joint:
            #     y_tmp = y + scene_origin.to(device=device)[:,None]
            #     for ii, ik in  enumerate(item_key):
            #         y_for_save[ik] = y_tmp[ii].cpu().data.numpy()

            total_num_sample += y.shape[0]
    if   TRAIN:
        scheduler.step()
        train_losses /= total_num_sample
        train_loss_joints /= total_num_sample
        train_loss_root /= total_num_sample
        train_losses=train_losses.cpu().item()
        train_loss_joints=train_loss_joints.cpu().item()
        train_loss_root=train_loss_root.cpu().item()
        lr = optimizer.param_groups[0]['lr']
        losses_str = f'{train_losses=:.5f} {train_loss_joints=:.5f} {train_loss_root=:.5f}'
        if 1:
            elapse = time.time() - t_s
            str1='====> Epoch: {} elapse: {:.2f} {} lr: {:.5f}'.format(epoch, elapse, losses_str, lr)
            print(str1)
            csv_dir = f'{result_dir}/err.csv'
            with open(csv_dir, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                from datetime import datetime
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                writer.writerow([
                    gethostname() ,
                    dt_string,
                    str1,
                ])
        if 1:
            dftb.set_cur_dic('lr', lr )
            dftb.set_cur_dic('elapse', elapse )
            dftb.set_cur_dic('train_losses', train_losses )
            dftb.set_cur_dic('train_loss_joints', train_loss_joints )
            dftb.set_cur_dic('train_loss_root', train_loss_root )
    else:
        path_err = path_err * 1000 / total_num_sample
        pose_err = pose_err * 1000 / total_num_sample
        all_err = all_err * 1000 / total_num_sample
        if t_pred == 60:
            log_idxs1 = [14, 29, 44, 59]
        elif t_pred == 30:
            log_idxs1 = [  14, 29]
        else:
            log_idxs1 = [int(i/4 * t_pred) for i in range(4)]
            log_idxs1[-1] = t_pred-1
            print(f"==>> log_idxs1: {log_idxs1}")
        log_idxs1 = np.array(log_idxs1)
        log_idxs = np.arange(t_pred)
        header = ['err'] + list(np.arange(t_pred)[log_idxs]) + ['mean']
        header1 = ['err'] + list(np.arange(t_pred)[log_idxs1]) + ['mean']
        csv_dir = f'{result_dir}/err.csv'
        with open(csv_dir, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            from datetime import datetime
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            writer.writerow([
                '\n',
                f"{result_dir=} {epoch=}",
                dt_string,
            ])
            # if header is not None:
            # write the header
            writer.writerow(header)

            data = ['path_err'] + list(path_err[log_idxs]) + [path_err.mean()]
            writer.writerow(data)
            data = ['joint_err'] + list(pose_err[log_idxs]) + [pose_err.mean()]
            writer.writerow(data)
            data = ['all_joint_err'] + list(all_err[log_idxs]) + [all_err.mean()]
            writer.writerow(data)

            writer.writerow(header1)
            data = ['path_err'] + list(path_err[log_idxs1]) + [path_err.mean()]
            writer.writerow(data)
            data = ['joint_err'] + list(pose_err[log_idxs1]) + [pose_err.mean()]
            writer.writerow(data)
            data = ['all_joint_err'] + list(all_err[log_idxs1]) + [all_err.mean()]
            writer.writerow(data)
        if 1:
            dftb.set_cur_dic('path_err_mean', path_err.mean() , print_=True )
            dftb.set_cur_dic('joint_err_mean', pose_err.mean() , print_=True )

        # if args.save_joint:
        #     np.savez_compressed(f'{cfg.result_dir}/prediction_{MODE}.npz', y=y_for_save)
    dftb.set_cur_dic('len(dataset)', len(dataset) ,  )
    if SAVE_csv_dftb:
        dftb.get_df().to_csv( csv_dftb  ) # cannot use mode='a'  )

if __name__ == '__main__':

    MODE = 'train' if TRAIN else 'test'

    """setup"""
    seed=0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


    from conf_torch import *
    """parameter"""
    over_all_step = 0



    """model"""
    model = pipelines.Pipeline()
    model.float()
    print(">>> total params: {:.5f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))
    
    model.to(device) # otherwise error when .step when load from cp
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                    eps=ADAM_eps ,
                )
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=num_epoch_fix, nepoch=num_epoch)


    if cp_iter != 0:
        if cp_iter=='AUTO' or cp_iter=='auto':
            cp_iter=get_latest_ckpt_B()[1]
            if  1  :
                user_confirm=input(f"auto load {cp_iter=}? (y/n): ")
                if user_confirm.lower()!='y':
                    exit(0)
        cp_path = model_path_trained % cp_iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location=device)
        optimizer.load_state_dict(model_cp['opt_dict'])
        scheduler.load_state_dict(model_cp['scheduler_dict'])
        model.load_state_dict(model_cp['model_dict'])
        del model_cp # otherwise consumes GPU memory

    """data"""
    dataset = dataset_cls(MODE, scene_seen_or_unseen=scene_seen_or_unseen )
    print(f">>> total sub sequences: {dataset.__len__()}")

    
    from tensorboardX import SummaryWriter
    from my_py_lib.miscellaneous.MyDataFrame import MyDataFrameA_withTB
    dftb = MyDataFrameA_withTB( SummaryWriter(  f'tb/{result_id}'  ) )
    csv_dftb = f'{result_dir}/df.csv'
    if SAVE_csv_dftb:   # if csv_dftb exists, rename it
        if os.path.exists(csv_dftb):
            _i = 1
            new_name_of_existing = f'{result_dir}/df-{_i}.csv'
            while os.path.exists(new_name_of_existing):
                _i += 1
                new_name_of_existing = f'{result_dir}/df-{_i}.csv'
            print(f"rename existing {csv_dftb} to {new_name_of_existing}")
            os.rename(csv_dftb, new_name_of_existing)
    
    if TRAIN:
        model.train()
        for i in range(cp_iter, num_epoch):
            if dataset is None:
                dataset = dataset_cls(MODE,  )
            train(i, TRAIN, dataset)
            if save_model_interval > 0 \
                    and (i + 1) % save_model_interval == 0 :
                cp_path = model_path % (i + 1)
                model_cp = {'model_dict':   model.state_dict(),
                            'opt_dict': optimizer.state_dict(),
                            'scheduler_dict': scheduler.state_dict()}
                print('saving model to checkpoint: %s' % cp_path)
                torch.save(model_cp,cp_path)
                print('model saved')
            if eval_gate(i,num_epoch,    eval_model_interval):
                dataset = None
                dataset_val = dataset_cls('test',)
                train(i, False, dataset_val)
                del dataset_val
    else:
        assert cp_iter>0
        train(cp_iter, TRAIN, dataset)
   
    
    
    
    
