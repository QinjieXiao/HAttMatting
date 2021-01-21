"""
    end to end network

Author: Zhengwei Li
Date  : 2018/12/24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .M_Net import M_net, M_tiny_net
from .T_Net import T_mv2_unet, RD_FPNnet


T_net = RD_FPNnet
M_net = M_tiny_net


class Model(pl.LightningModule):
    '''
		end to end net 
    '''

    def __init__(self):

        super(Model, self).__init__()

        self.t_net = T_net()
        self.m_net = M_net()



    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        trimap_softmax = F.softmax(trimap, dim=1)

        # paper: bs, fs, us
        bg, fg, unsure = torch.split(trimap_softmax, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p

    def migrate(self, state_dict):
        with torch.no_grad():
            for i, (name, p) in enumerate(self.state_dict().items()):
                if name in state_dict:
                    if p.data.shape == state_dict[name].shape:
                        print(i, name)
                        p.copy_(state_dict[name])






