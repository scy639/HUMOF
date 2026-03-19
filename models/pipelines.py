import torch.nn.functional
from conf import *
from torch import nn
from torch import Tensor
from . import pf_extractors 
from . import transformers 
from . import positionEncodings 
from utils import *
from my_py_lib.misc_util import to_ndarray
import globals_
from .dct_scale import l__dct_scale_4_dim2
from . import gcn

pad_idx = tuple(   list(range(t_his)) + [t_his-1]*t_pred  )
class Preprocess:
    @classmethod
    def A(
        cls,     
        #from dataloader
        pose, objs_pc, scene_origin,objs_semId ,  item_key,
    ):
        if 1:
            bs=pose.shape[0]
            num_obj = objs_pc.shape[1]
            nj = pose.shape[2]
            BO = bs*num_obj
        with torch.no_grad():
            objs_pc=objs_pc.to(device=device)
            objs_semId=objs_semId.to(device=device)
            # (B, O,...) -> (B*O,...)
            objs_pc_bsFlat= objs_pc.view(-1, *objs_pc.shape[2:])
            # BO: num_obj_bsFlat, treated as new B new O
            
            assert bs == pose.shape[0]
            npts = MAX_OBJ_PC_NUMBER

            joints = pose.to(device=device)

            # is_cont = (objs_pc_bsFlat[:, None, :, None, :] - joints[:, :, None, :, :]).norm(dim=-1)
            """
            left:   B,O,0,      N,0, 3 
            right:  B,0,45==T,  0,21,3
            t1:     B,O,45==T,  N,21,3
            """
            t1=objs_pc[:,:, None, :, None, :] - joints[:, None,:, None, :, :] 
            t2=t1.view(   BO, t_total, npts, nj, 3  ) # B,O,... -> BO,...
            is_cont=t2.norm(dim=-1)
            
            is_cont_gauss = torch.exp(-0.5*is_cont**2/sigma**2)

            joints_orig = joints[:, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
            joints = joints - joints_orig
            joints[:, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX+1] = joints_orig
            #joints: B,T,J,3

            is_cont_pad = is_cont_gauss[:, pad_idx].reshape([BO, t_his + t_pred, -1])
            is_cont_dct = torch.matmul(DCT_IDCT.dct_m[None], is_cont_pad).reshape([BO, dct_n, npts, nj])
            is_cont_dct = is_cont_dct.permute(0, 1, 3, 2).reshape([BO, dct_n * nj, npts])
            
            joints_pad=joints[:,pad_idx] # !!!  
            if  1 :
                #
                joints_repeat_O=joints.tile(  num_obj,1,1,1  )# [B*O, t_total, 21, 3]
                assert joints_repeat_O.shape[0]==BO
                # -> [..., 21*3]
                joints_repeat_O=joints_repeat_O.view(  *joints_repeat_O.shape[:2],-1)# BO,T,63
                dcts= DCT_IDCT.perform_DCT(  x=joints_pad.view(bs,t_total,-1), )# B,DCT,J*3
                dcts=dcts.permute(0,2,1)
        return joints,\
                joints_repeat_O,\
                dcts,\
                objs_pc_bsFlat,\
                objs_semId,\
                is_cont_dct

    @classmethod
    def A_multiPerson(cls,_tmp):
        _tmp2 = _tmp[:-1]
        if CHECK_:
            print_randomly(f"{_tmp=}", 0.01 , )
        others = _tmp[-1]
        _tmp2 = cls.A( *_tmp2 )
        return (
            * _tmp2,
            others,
        )
    @classmethod
    def B(cls,batch):
        _tmp = torch.utils.data.default_collate(batch)
        return *_tmp,   * cls.A(*_tmp)
    @classmethod
    def truncateA(cls,batch):
        INDEX_OF_pc=1
        INDEX_OF_semId=3
        INDEX_OF_others=5
        if MULTI_PERSON_MODE:
            INDEX_OF_pc=INDEX_OF_others
        l_objs_pc=[single_dataitem_tu[INDEX_OF_pc] for single_dataitem_tu in batch]
        l_objs_semId=[single_dataitem_tu[INDEX_OF_semId] for single_dataitem_tu in batch]
        # for 'batch[i][1]=' below. otherwise got "'tuple' object does not support item assignment"
        for i in range(len(batch)):
            batch[i]=list(batch[i])
        min_obj_num=min([single_objs_pc.shape[0] for single_objs_pc in l_objs_pc])
        for i,objs_pc in enumerate(l_objs_pc):
            if objs_pc.shape[0]>min_obj_num:
                remain_obj_idxs= random.sample(range(objs_pc.shape[0]), min_obj_num)
                batch[i][INDEX_OF_pc]=objs_pc[remain_obj_idxs]
                batch[i][INDEX_OF_semId]=l_objs_semId[i][remain_obj_idxs]
        batch = torch.utils.data.default_collate(batch)
        return batch
class Pipeline(nn.Module):
    semId_2_clip:dict=None
    @classmethod
    def _init_A(cls, device='cuda'):
        if INTRODUCE_semantic_label:
            l_clip_npy = CLIP_DIR.glob(f'*.npy')
            cls.max_semId = max(int(clip_path.stem) for clip_path in l_clip_npy)
            clip_features = torch.zeros((cls.max_semId + 1, 512), device=device)   
            for clip_path in l_clip_npy:
                semId = int(clip_path.stem)
                clip_feature = np.load(clip_path)
                clip_feature = torch.from_numpy(clip_feature).to(device=device).float()
                clip_features[semId] = clip_feature
            cls.semId_2_clip = clip_features
    def __init__(self,  ):
        super().__init__()
        if GCN_encode_ego or DEEP_SUPERVISION:
            if 1:
                GCN_dim__input_feature = 3*dct_n
                GCN_dim__hidden_feature = 256 # ~4x input_feature
                GCN__node_n = NUM_J
            else:
                GCN_dim__input_feature = dct_n
                GCN_dim__hidden_feature = 96 # ~4x input_feature
                GCN__node_n = NUM_J * 3
        if GCN_encode_ego:
            self.gcnQ1 = gcn.GCN(
                input_feature=GCN_dim__input_feature,
                hidden_feature=GCN_dim__hidden_feature,
                p_dropout=0.3,
                num_stage=GCN__num_stage,
                node_n=GCN__node_n,
            )
            self.gcnQ2 = gcn.GCN(
                input_feature=GCN_dim__input_feature,
                hidden_feature=GCN_dim__hidden_feature,
                p_dropout=0.3,
                num_stage=GCN__num_stage,
                node_n=GCN__node_n,
            )
        elif DEEP_SUPERVISION:
            self.gcnQ2 = gcn.GCN(
                input_feature=GCN_dim__input_feature,
                hidden_feature=GCN_dim__hidden_feature,
                p_dropout=0.3,
                num_stage=GCN__num_stage,
                node_n=GCN__node_n,
            )
        self.pf= pf_extractors.A()
        if INTRODUCE_semantic_label:
            self.mlp_clip  = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                # nn.Dropout(DROPOUT),
                nn.Linear(256, 256),
                nn.ReLU(),
                # nn.Dropout(DROPOUT),
                nn.Linear(256, 120)
            )   
        self.position_encoding = positionEncodings.LearnablePositionEncoding_A(
            dim=3 * dct_n,
            length=NUM_J,# J
        )
        if 1:
            dim_q=3*dct_n
            # dim_kv= dim_objNode
            num_heads_sa=4
            num_heads_ca=4
            # ff_dim=3*dct_n  *4  
            #
            tl0 = transformers.TransformerLayer(dim_q, 512, num_heads_sa, num_heads_ca, ff_dim=2048, )
            tl1 = transformers.TransformerLayer(dim_q, 512, num_heads_sa, num_heads_ca, ff_dim=2048, )
            tl2 = transformers.TransformerLayer(dim_q, 512, num_heads_sa, num_heads_ca, ff_dim=2048, )
            #
            tl3 = transformers.TransformerLayer(dim_q, 256, num_heads_sa, num_heads_ca, ff_dim=1024, dropout=2*DROPOUT )
            tl4 = transformers.TransformerLayer(dim_q, 256, num_heads_sa, num_heads_ca, ff_dim=1024, dropout=2*DROPOUT )
            #
            tl5 = transformers.TransformerLayer(dim_q, 128, num_heads_sa, num_heads_ca, ff_dim=512, dropout=2*DROPOUT )
            self.layers = nn.ModuleList([
                tl0,
                tl1,
                tl2,
                tl3,
                tl4,
                tl5,
            ])
            if 1:
                self.l__dct_scale_4_dim2 = torch.stack( l__dct_scale_4_dim2, dim=0)
                self.l__dct_scale_4_dim2 = nn.Parameter(
                    self.l__dct_scale_4_dim2.clone().detach(), 
                    requires_grad=True,
                )
            
            
            if MULTI_PERSON_MODE:
                if GCN_oth:
                    self.gcnQ1_oth = gcn.GCN(
                        input_feature=GCN_dim__input_feature,
                        hidden_feature=GCN_dim__hidden_feature,
                        p_dropout=0.3,
                        num_stage=GCN__num_stage,
                        node_n=GCN__node_n,
                    )
                if HHI_B:
                    self.C_hhi = C_hhi = 2
                else:
                    self.C_hhi = C_hhi = 3
                dim_othNode0 = dct_n*(NUM_J*C_hhi) + 8 
                if ENCODE_others_locally:
                    dimReservedFor_othGlo = 3*dct_n
                else:
                    dimReservedFor_othGlo = 0
                self.mlp_oth_to_512 = nn.Sequential(
                    nn.Linear(dim_othNode0, 512 - 8-dimReservedFor_othGlo ),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                )   
                if not (OL_cat_perJtoken_in_last_layers and NO_perP_token_in_last_layers):
                    self.mlp_oth_512_to_256 = nn.Sequential(
                        nn.Linear( 512 - 8 , 256 - 8 ),
                        nn.ReLU(),
                        nn.Dropout(DROPOUT),
                    )   
                    self.mlp_oth_256_to_128 = nn.Sequential(
                        nn.Linear( 256 - 8 , 128 - 8 ),
                        nn.ReLU(),
                        nn.Dropout(DROPOUT),
                    )   
                if OL_cat_perJtoken_in_last_layers:
                    self.mlp_oth_perJ_to_256 = nn.Sequential(
                        nn.Linear( dct_n*4 , 256 - 8 ),
                        nn.ReLU(),
                        nn.Dropout(DROPOUT),
                    )  
                    self.mlp_oth_perJ_to_128 = nn.Sequential(
                        nn.Linear( dct_n*4 , 128 - 8 ),
                        nn.ReLU(),
                        nn.Dropout(DROPOUT),
                    )  
                if 0:
                    self.other_transformer = transformers.SA_Transformer(
                    L=1,
                    dim_q= dim_othNode1,
                    num_heads_sa=4,
                    ff_dim= 2048  ,
                    dropout=DROPOUT,
                )
                if ENCODE_others_locally:
                    if not OL_SHARE_primary_SA:
                        self.other_encoder_transformer = transformers.SA_Transformer(
                        L=2,
                        dim_q= 3*dct_n,
                        num_heads_sa=4,
                        ff_dim= 256  ,
                        dropout=DROPOUT,
                    )
                    if 1:
                        self.global_token = nn.Parameter(torch.zeros(1, 1, 3*dct_n), requires_grad=True)
                        nn.init.normal_(self.global_token, std=0.02)
        self._init_A()
    def forward(self, 
                x0 :Tensor, # T,B*O,J*3  will be directly passed to pf w/o process
                dcts :Tensor, # B,J*3,DCT
                #
                scene :Tensor, # B*O,3,N  will be directly passed to pf w/o process
                #
                cont_dct :Tensor,  # will be directly passed to pf w/o process
                #
                objs_semId: Tensor,   # B,O
                # joints (  time domain
                primary: Tensor, # B,  T,J,3
                others: Tensor, #  B,P,T,J,3
                ):#-> # B,J*3,DCT
        bs=dcts.shape[0]
        BO=scene.shape[0]
        O=BO//bs
        J=dcts.shape[1]//3
        DCT=dct_n
        if globals_.cur_epoch >= LEARN_ds_since:
            self.l__dct_scale_4_dim2.requires_grad = True
        else:
            self.l__dct_scale_4_dim2.requires_grad = False
        if not MULTI_PERSON_MODE:
            del others
        if  MULTI_PERSON_MODE:
            C_hhi = self.C_hhi
            P = others.shape[1]
            BP = bs*P
            if CHECK_:
                assert others.shape==(bs,P,t_his,J,3),others.shape
            primary_root = primary[:,  :,ROOT_JOINT_IDX]                  # B,  t_his,  3
            others_root =  others [:,:,:,ROOT_JOINT_IDX]                  # B,P,t_his,  3
            others_root_ =  others [:,:,:,ROOT_JOINT_IDX:ROOT_JOINT_IDX+1] # B,P,t_his,1,3
            if ENCODE_others_locally:
                with torch.no_grad():
                    others_pad=others[:,:,pad_idx] # B,P,t_total,J,3
                    dct_others = DCT_IDCT.perform_DCT(  x=others_pad.view(bs*P,t_total,J*3 ), )#BP,dct,J*3
                    dct_others = dct_others.permute(0, 2, 1)  # BP,J*3,dct,
                    dct_others = dct_others.reshape(bs*P, J, 3*dct_n)
                if GCN_oth:
                    dct_others = self.gcnQ1_oth(dct_others)
                global_token = self.global_token.repeat(bs*P,1,1)
                if CHECK_:   assert global_token.shape==(bs*P,1,3*dct_n)
                dct_others = self.position_encoding(dct_others)
                _dct_others = torch.cat([global_token, dct_others], dim=1)
                if CHECK_:   assert _dct_others.shape==(bs*P,1+J,3*dct_n)
                if not OL_SHARE_primary_SA:
                    _dct_others += self.other_encoder_transformer(_dct_others)
                else:
                    for i_layer,layer in enumerate( self.layers[ : 3 ] ):
                        if   0  :
                            if CHECK_: assert len(self.l__dct_scale_4_dim2)==len(self.layers)
                            dct_scale_4_dim2 = self.l__dct_scale_4_dim2[i_layer]
                        else:
                            dct_scale_4_dim2 = None
                        _dct_others = layer.forward_without_CA(_dct_others,
                                        dct_scale_4_dim2, None, None, None)
                dct_others = _dct_others[:, 1:, :]  # bs*P,J,3*dct_n
                dct_others = dct_others.view(bs,P,J,3*dct_n)
                global_token = _dct_others[:, 0, :]  # bs*P,1,3*dct_n
                del _dct_others
                global_token = global_token.view(bs,P,3*dct_n)
            if HHI_B:
                #                (B, 0,    T, J, 0,    3 ) -       (B,P,T,0,   J,3)
                xyzDis0 = primary[:, None, :, :, None, : ] - others[:,:,:,None,:,:] # B,P,T,J(primary),J,3
                if CHECK_: assert xyzDis0.shape==(bs,P,t_his,J,J,3)
                dis0 = xyzDis0.norm(dim=-1)    # B,P,T,J(primary),J
                dis0 = dis0.min(dim=-1).values # B,P,T,J(primary)
                if CHECK_: assert dis0.shape==(bs,P,t_his,J)
                #                (B, 0,    T, 0,    J , 3 ) -       (B,P,T,J,0,   3)
                xyzDis1 = primary[:, None, :, None, : , : ] - others[:,:,:,:,None,:] # B,P,T,J(others),J,3
                if CHECK_: assert xyzDis1.shape==(bs,P,t_his,J,J,3)
                dis1 = xyzDis1.norm(dim=-1)    # B,P,T,J(others),J
                dis1 = dis1.min(dim=-1).values # B,P,T,J(others)
                if CHECK_: assert dis1.shape==(bs,P,t_his,J)
                t2= torch.cat([dis0,dis1,],dim=-1) # B,P,t_his,J*C_hhi
                if CHECK_: assert t2.shape==(bs,P,t_his,J*2), t2.shape
                if OL_cat_perJtoken_in_last_layers:
                    others_vector_perJ = dis1.view( bs, P, t_his, J )
                    others_vector_perJ = torch.exp(-0.5*others_vector_perJ**2/sigma**2)
                    others_vector_perJ = others_vector_perJ[:, :, pad_idx]
                    if CHECK_:  assert others_vector_perJ.shape==(bs,P, t_total, J)
                    others_vector_perJ = DCT_IDCT.perform_DCT(  others_vector_perJ ) # B,P,DCT,J
                    others_vector_perJ = others_vector_perJ.permute( 0,1,3,2 ) # B,P,J,DCT
                    if CHECK_:  assert others_vector_perJ.shape == ( bs,P,J,dct_n )
                del dis0,dis1
            else:
                """
                get 3 xyz distances.      B,P,t_his,J,3
                    primary to others, per-joint.          
                    primary's root to others each joint 
                    others's root to primary each joint 
                """
                xyzDis0 = primary[:, None] - others
                xyzDis1 = primary[:, None, :,
                                ROOT_JOINT_IDX:ROOT_JOINT_IDX+1] - others
                xyzDis2 = others_root_ - primary[:, None]
                if CHECK_:
                    #TODO check primary root at origin
                    assert (bs,P,t_his,J,3)==xyzDis0.shape==xyzDis1.shape==xyzDis2.shape, \
                            f"{xyzDis0.shape},{xyzDis1.shape},{xyzDis2.shape}"
                # 3 distances (xyz-> L2 dist).  B,P,t_his,J
                dis0 = xyzDis0.norm(dim=-1)
                dis1 = xyzDis1.norm(dim=-1)
                dis2 = xyzDis2.norm(dim=-1)
                # dct
                t2= torch.cat([dis0,dis1,dis2],dim=-1) # B,P,t_his,J*3. 3 means dis012, not xyz
                del dis0,dis1,dis2
            t2 = t2.view((bs, P, t_his, J*C_hhi))
            is_cont = t2
            is_cont_gauss = torch.exp(-0.5*is_cont**2/sigma**2)
            is_cont_pad = is_cont_gauss[:, :, pad_idx]
            if CHECK_:
                assert is_cont_pad.shape==(bs,P, t_total, J*C_hhi)
            is_cont_dct = DCT_IDCT.perform_DCT(  is_cont_pad )# B,P,DCT,J*C. C means dis012, not xyz
            is_cont_dct = is_cont_dct.view(  (bs,P,dct_n*J*C_hhi)  )
            others_vector = is_cont_dct
            del t2,is_cont,is_cont_gauss,is_cont_dct,is_cont_pad
            if 1: # others_PE
                others_root_lastFrame=others_root[:, :, -1]   # B,P,C
                others_dis_lastFrame=others_root_lastFrame.norm(dim=-1) # B,P. dis to origin (root of primary)
                others_exp_root_lastFrame= torch.exp(-0.5*others_root_lastFrame**2/sigma_4pe**2) 
                others_exp_dis_lastFrame = torch.exp(-0.5*others_dis_lastFrame**2/sigma_4pe**2)
                others_PE = torch.cat([
                    others_root_lastFrame,others_exp_root_lastFrame, # B,P,C
                    others_dis_lastFrame[:,:,None],others_exp_dis_lastFrame[:,:,None], # B,P,1
                ],dim=-1) # B,P,8
                others_vector=torch.cat([
                    others_vector,
                    others_PE,
                ],dim=-1) # B, P, dim_othNode0= DCT*J*C + 8
                others_vector_512 = self.mlp_oth_to_512( others_vector ) # B,P,512-8-dimReservedFor_othGlo
                if ENCODE_others_locally:
                    others_vector_512 = torch.cat([
                        others_vector_512,
                        global_token,
                    ],dim=-1) # B, P, 512-8
                others_vector_512=torch.cat([
                    others_vector_512,
                    others_PE,
                ],dim=-1) # B, P, 512
                if not (OL_cat_perJtoken_in_last_layers and NO_perP_token_in_last_layers):
                    others_vector_256 = self.mlp_oth_512_to_256( others_vector_512 ) # B,P,256-8
                    others_vector_128 = self.mlp_oth_256_to_128( others_vector_256 ) # B,P,128-8
                    others_vector_256=torch.cat([
                        others_vector_256,
                        others_PE,
                    ],dim=-1) # B, P, 256
                    others_vector_128=torch.cat([
                        others_vector_128,
                        others_PE,
                    ],dim=-1) # B, P, 128
                if OL_cat_perJtoken_in_last_layers:
                    others_vector_perJ = torch.cat([
                        others_vector_perJ.reshape( bs,P*J,dct_n ), # bs,P*J,dct_n
                        dct_others.reshape( bs, P*J, 3*dct_n ), # bs,P,J,3*dct_n
                    ],dim= -1 )# bs,P*J,4*dct_n
                    others_vector_perJ_256 = self.mlp_oth_perJ_to_256( others_vector_perJ )# bs,P*J,256-8
                    others_vector_perJ_128 = self.mlp_oth_perJ_to_128( others_vector_perJ )# bs,P*J,128-8
                    del others_vector_perJ
                    # ------- perJ PE --------
                    if 0: # each joint use its own PE
                        pass
                    else:# toBetter all joints use this person's PE
                        others_perJ_PE = others_PE.repeat(1, J, 1)
                    if CHECK_: assert others_perJ_PE.shape == (bs,P*J,8)
                    others_vector_perJ_256 = torch.cat([
                        others_vector_perJ_256,
                        others_perJ_PE,
                    ],dim= -1 ) # B, P*J, 256
                    others_vector_perJ_128 = torch.cat([
                        others_vector_perJ_128,
                        others_perJ_PE,
                    ],dim= -1 ) # B, P*J, 128
                    del others_perJ_PE
                    #
                    if not NO_perP_token_in_last_layers:
                        others_vector_256 = torch.cat([
                            others_vector_256,
                            others_vector_perJ_256,
                        ],dim= 1 ) # B, P+P*J, 256
                        others_vector_128 = torch.cat([
                            others_vector_128,
                            others_vector_perJ_128,
                        ],dim= 1 ) # B, P+P*J, 128
                    else:
                        others_vector_256 = others_vector_perJ_256 # B, P*J, 256
                        others_vector_128 = others_vector_perJ_128 # B, P*J, 128
                    del others_vector_perJ_256,others_vector_perJ_128
                others_vectors = [others_vector_512]*3 \
                                +[others_vector_256]*2 \
                                +[others_vector_128]*1
            if 0:
                others_vector = self.other_transformer( others_vector ) 
        if NORM_OBJ:
            objs_center_flat=scene.mean(dim=2)
            assert (BO,3,)==objs_center_flat.shape
            scene=scene-objs_center_flat[:,:,None] # have tested
        if   CHECK_  :
            assert    J==NUM_J
            if INTRODUCE_semantic_label:
                assert objs_semId.shape==(bs,O)
        
        pf1_list, coords_list = self.pf(  x0, scene, cont_dct= cont_dct   )# B*O,C1
        
        
        q=dcts.reshape(  (bs,J,-1)  ) # B,J,3*DCT
        if GCN_encode_ego:
            q = self.gcnQ1(q)
        q = self.position_encoding(q) 
        # context=pf1 # B,O,C1
        contexts=[None]*6
        if INTRODUCE_semantic_label:
            objs_semFeature=self.sem_id2feature(objs_semId.view(bs*O)   )  # B*O,C2
            objs_semFeature=objs_semFeature.view(bs,O,-1)
            context=torch.cat([context,objs_semFeature],dim=-1) # B,O,C1+C2
        if CONCAT_OBJ_PE:
            iLayer_2_iSap={
                0:0,
                1:0,
                2:0,
                3:1,
                4:1,
                5:2,
            }
            for i in range(6):
                iSap = iLayer_2_iSap[i]
                pf1=pf1_list[iSap]
                objs_center= coords_list[iSap]
                O=pf1.shape[1]
                assert objs_center.shape==(bs,O,3)  ,  objs_center.shape
                """
                distance from obj center to origin. Euclidean distance
                norm: sum(abs(x)**ord)**(1./ord), ord by default is 2
                """
                objs_dis=objs_center.norm(dim=-1) # B,O
                #
                objs_exp_center= torch.exp(-0.5*objs_center**2/sigma_4pe**2) 
                objs_exp_dis = torch.exp(-0.5*objs_dis**2/sigma_4pe**2) 
                contexts[i]=torch.cat([
                    pf1,# B,O, C
                    objs_center,objs_exp_center, # B,O, 3;
                    objs_dis[:,:,None],objs_exp_dis[:,:,None], # B,O, 0;
                ],dim=-1) # B,O, C+8
                if MULTI_PERSON_MODE:
                    contexts[i] = torch.cat([contexts[i], others_vectors[i]], dim=1) # B,O+P,C+8
                del O
        if 0:
            context= self.obj_transformer( context ) 
        q_from_inter_layers = []
        if 1:
            q_=q
            for i_layer,layer in enumerate( self.layers ):
                if  globals_.cur_epoch >= APPLY_ds_since:
                    if CHECK_:
                        assert len(self.l__dct_scale_4_dim2)==len(self.layers)
                    dct_scale_4_dim2 = self.l__dct_scale_4_dim2[i_layer]
                else:
                    dct_scale_4_dim2 = None
                context = contexts[i_layer]
                if DROPOUT_ctx_token.enable:
                    if i_layer > 2:
                        if self.training:
                            mask = torch.rand(context.shape[1], device=context.device) > DROPOUT_ctx_token.prob
                            contexts_dropped = context[:, mask, :]
                            del mask
                        else:
                            contexts_dropped = context
                        context = contexts_dropped
                        del contexts_dropped
                q_ = layer(q_, context, dct_scale_4_dim2, None, None, None)
                if DEEP_SUPERVISION:
                    if i_layer==2:
                        q_from_inter_layers.append(q_)
        if GCN_encode_ego:
            q_ = self.gcnQ2(q_)
        q_ = q_ +q #(res cross GCN decoder)
        dcts_pred=q_.view(  (bs, J*3,DCT)  )# same as dcts
        for i in range(len(q_from_inter_layers)):
            q_from_inter_layers[i] = self.gcnQ2( q_from_inter_layers[i] )
            q_from_inter_layers[i] = q_from_inter_layers[i] + q
            q_from_inter_layers[i] = q_from_inter_layers[i].view(  (bs, J*3,DCT)  )
        return dcts_pred,q_from_inter_layers