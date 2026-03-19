# import root_config
import json
import os
import numpy as np
import torch
import pandas as pd


class MyDataFrameA:
    def __init__(self):
        self.cur_k = None
        self.k2dic = {}
    def new_k(self,new_k):
        """
        this k corresponds to the index (row)   in df, can be multi-level index(if new_k is tuple)
        """
        assert new_k not in self.k2dic,(new_k,self.k2dic)
        self.k2dic[new_k]={}
        self.cur_k=new_k
    def set_or_new_k(self,new_k):
        if new_k not in self.k2dic:
            self.new_k(new_k)
        else:
            self.cur_k=new_k
    def set_cur_dic(self,k,v):
        """
        cur_dic means k2dic[cur_k], not k2dic
        
        this k corresponds to the colume name in df
        """
        self.k2dic[self.cur_k][k]=v
    def get_cur_dic(self,k,):
        return self.k2dic[self.cur_k][k]
    def get_df(self):
        df=pd.DataFrame.from_dict(self.k2dic, orient='index')
        # df = df.fillna(None)  # Set default value to None instead of NaN
        return df
    def clear(self):
        self.cur_k = None
        self.k2dic = {}


class MyDataFrameA_withTB(MyDataFrameA):
    def __init__(self, tb):
        """
        tb: SummaryWriter.  from tensorboardX import SummaryWriter
        """
        super().__init__()
        self.tb = tb

    def new_k(self, new_k):
        assert isinstance(new_k, int),\
        "k must be an integer because it will be x-axis in tb"
        super().new_k(new_k)

    def set_cur_dic(self, k, v , print_=False):
        super().set_cur_dic(k, v)
        # add_scalar params:  tag, scalar_value, global_step
        self.tb.add_scalar(
            f"{k}", 
            v, self.cur_k, 
        )
        if print_:
            print(f"{k}: {v}")