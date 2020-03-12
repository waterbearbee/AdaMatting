import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self,c,out_c,k=(7,7)): # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0],1), padding =(int((k[0]-1)/2),0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k[0]), padding =(0,int((k[0]-1)/2)))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k[1]), padding =(0,int((k[1]-1)/2)))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1],1), padding =(int((k[1]-1)/2),0))
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        
        x = x_l + x_r
        
        return x