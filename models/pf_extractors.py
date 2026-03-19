import torch.nn.functional
from conf import *
from torch import nn
from pvcnn.models.prox.pvcnnpp import PVCNN2_SA, PVCNN2_FP


"""
modify from PVCNN2_DCT_CONT
"""
class A(nn.Module):
    def __init__(self,  ):
        super().__init__()
        self.input_dim = input_dim = 3+NUM_J*dct_n
        self.dct_n = dct_n  
        self.out_dim = out_dim = input_dim
        # self.aux_dim = aux_dim = model_specs['aux_dim']
        # self.nh_rnn = nh_rnn = model_specs['nh_rnn']
        self.num_classes = num_classes = NUM_J
        self.is_bn = is_bn = True


        """
        self.sa_blocks = sa_blocks = [ # pfSA_A: 823,826's version (I just re-format it)
            [[64 , 2, 32  ],  [1024, 0.1, 32, [64  , 128       ]]],
            [[128, 2, 16  ],  [256 , 0.2, 32, [128 , 128       ]]],
            [[128, 2, 8   ],  [64  , 0.4, 32, [128 , 256       ]]],
            [[            ],  [16  , 0.8, 32, [256 , 256 , 512 ]]],
            #
            [[            ],  [4   , 1.6, 32, [512 , 512 , 1024]]],
            [[            ],  [1   , 3.2, 32, [1024, 1024, 2048]]],
        ]
        """
        """
        self.sa_blocks = sa_blocks = [ # pfSA_B
            [[64 , 2, 32  ],  [256, 0.1, 32, [64  , 128       ]]],
            [[128, 2, 16  ],  [64 , 0.2, 32, [128 , 128       ]]],
            [[128, 2, 8   ],  [16  , 0.4, 32, [128 , 256       ]]],
            [[            ],  [4  , 0.8, 32, [256 , 256 ,   ]]],
            [[            ],  [1   , 1.6, 32, [256 , 256 , 512]]],
        ]
        """
        self.sa_blocks = sa_blocks = [   # pfSA_H2
            [[64 , 2, 32  ],  [256, 0.1, 32, [64  , 128-8        ]]],
            [[128, 2, 16  ],  [64 , 0.2, 32, [128-8 , 128-8       ]]],
            [[128, 2, 8   ],  [16 , 0.4, 32, [128-8 , 256-8       ]]],
            [[            ],  [4  , 0.8, 32, [256-8 , 256-8 ,512-8 ]]],
        ]
        
        self.fp_blocks = fp_blocks = [[[256, 256], [256, 1, 8]],
                                       [[256, 256], [256, 1, 8]],
                                       [[256, 128], [128, 2, 16]],
                                       [[128, 128, 64], [64, 1, 32]]]

        # # encode input human poses
        # self.x_enc = nn.Linear(input_dim+aux_dim, nh_rnn)
        # self.x_gru = nn.GRU(nh_rnn,nh_rnn)

        # encode scene point cloud
        self.pointnet_sa = PVCNN2_SA(extra_feature_channels=num_classes*dct_n,num_classes=num_classes,
                               sa_blocks=sa_blocks, fp_blocks=fp_blocks,with_classifier=False,is_bn=is_bn)

        # self.pointnet_fp = PVCNN2_FP(extra_feature_channels=num_classes*dct_n,num_classes=num_classes,
        #                        sa_blocks=sa_blocks, fp_blocks=fp_blocks,with_classifier=False,is_bn=is_bn,
        #                        sa_in_channels=self.pointnet_sa.sa_in_channels,
        #                        channels_sa_features=self.pointnet_sa.channels_sa_features+nh_rnn)

        # self.out_mlp = nn.Sequential(nn.Linear(fp_blocks[-1][-1][0],nh_rnn),
        #                                 nn.Tanh(),
        #                                 nn.Linear(nh_rnn,num_classes*dct_n))

    def forward(self, x, scene, aux=None, cont_dct=None):
        """
        x: [seq,bs,dim]
        aux: [bs, seq, dim]
        scene: [bs, 3, N]
        cont_dct: [bs, 21*dct_n, sn]  #: dct_n=20; sn==N
        """
        bs, _, npts = scene.shape

        # # encode x
        # if aux is not None:
        #     hx = torch.cat([x, aux.transpose(0,1)],dim=-1)
        # else:
        #      hx = x
        # hx = self.x_enc(hx)
        # hx = self.x_gru(hx)[1][0] # [bs, dim]

        #each point in the scene has these 'attributes'(channel)：coord; dis to each joint (in dct). paper (contAware): "DCT coefficients of all joints as its feature"
        hs = torch.cat([scene, cont_dct], dim=1)

        coords_list, coords, features, in_features_list = self.pointnet_sa(hs)
        if 1:
            coords_list.append(coords)
            in_features_list.append(features)
            coords_list:list = coords_list[-3:]
            in_features_list:list = in_features_list[-3:]
            # for each item in in_features_list and coords_list, permute(0,2,1)
            for i in range(len(in_features_list)):
                in_features_list[i] = in_features_list[i].permute(0,2,1)
                coords_list[i] = coords_list[i].permute(0,2,1)
            in_features_list.reverse()
            coords_list.reverse()
            return in_features_list, coords_list
        ret=features.view(bs,-1) # features: (bs,512,16==num_centers ) 
        if SAP_as_objNode:
            coords_ret=coords.view(bs,-1)
            return ret,coords_ret
        return ret
        features = torch.cat([features, hx[:, :, None].repeat([1, 1, features.shape[-1]])], dim=1)
        hfp = self.pointnet_fp(coords_list, coords, features, in_features_list)  # [bs,dim,npts]

        hc = self.out_mlp(hfp.transpose(1, 2)).transpose(1, 2)
        hc = hc + cont_dct
        return hc