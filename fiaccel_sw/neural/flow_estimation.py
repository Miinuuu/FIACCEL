import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
    
class flow_estimation(nn.Module):
    def __init__(self, 
                 depths = [8, 16, 32 , 64, 32 ,16 ,8],
                 ):
        super().__init__()
       

        self.enc =torch.nn.Sequential(  nn.Conv2d(3, depths[0], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU(),

                                        nn.Conv2d(depths[0], depths[1], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU(),

                                        nn.Conv2d(depths[1], depths[2],  stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU())

        self.aspp1 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=1, kernel_size=(3,3), padding=(1,1), bias=False),
                                        nn.ReLU(),)

        self.aspp2 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,2), kernel_size=(3,3), padding=(1,2), bias=False),
                                        nn.ReLU(),)
                                        
        self.aspp3 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,3), kernel_size=(3,3), padding=(1,3), bias=False),
                                        nn.ReLU(),)
        self.aspp4 = torch.nn.Sequential(  nn.Conv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,4), kernel_size=(3,3), padding=(1,4), bias=False),
                                        nn.ReLU(),)

        self.shrink = torch.nn.Sequential(  nn.Conv2d(depths[3]*4, depths[3], stride=(1, 1), dilation=1, kernel_size=(1,1), padding=(0,0), bias=False),
                                            nn.ReLU(),
        )

        self.dec =torch.nn.Sequential (
                                nn.ConvTranspose2d(depths[3], depths[4], 3, stride=2, padding=1, bias=False, output_padding=1),nn.ReLU(),
                                nn.ConvTranspose2d(depths[4], depths[5], 3, stride=2, padding=1, bias=False, output_padding=1),nn.ReLU(),
                                nn.ConvTranspose2d(depths[5], depths[6],  3, stride=2, padding=1, bias=False, output_padding=1),nn.ReLU()
        )

        self.output = torch.nn.Sequential(  nn.Conv2d(depths[6], 5, stride=(1, 1), kernel_size=(3,3), padding=1, bias=False),
                                            nn.Tanh()
        )         
    def bottleneck(self,FM) :
        fm1= self.aspp1(FM)
        fm2= self.aspp2(FM)
        fm3= self.aspp3(FM)
        fm4= self.aspp4(FM)
        return self.shrink(torch.cat([fm1,fm2,fm3,fm4],dim=1))

    def forward(self,img0,img1, timestep=0.5):
        
        B,C,H,W= img0.shape
        img_cat = torch.cat([img0,img1],dim=0)

        encoder_FM = self.enc(img_cat)
        encoder_FM0=encoder_FM[:B,...]
        encoder_FM1=encoder_FM[B:,...]
        encoder_FM =  torch.cat([encoder_FM0,encoder_FM1],dim=1)
        bottleneck_FM = self.bottleneck(encoder_FM)
        dec_FM = self.dec(bottleneck_FM)
        out =    self.output(dec_FM)
        
        del img_cat
        del encoder_FM
        del encoder_FM0
        del encoder_FM1
        del bottleneck_FM
        del dec_FM

        return out


import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor
from brevitas.nn import QuantIdentity

class flow_estimation_int(nn.Module):
    def __init__(self, 
                 depths = [8, 16, 32 , 64, 32 ,16 ,8],
                 ):
        super().__init__()
       
        self.quant_input = QuantIdentity(bit_width=8, return_quant_tensor=True)

        self.enc =torch.nn.Sequential(  qnn.QuantConv2d(3, depths[0], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU(),

                                        qnn.QuantConv2d(depths[0], depths[1], stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU(),

                                        qnn.QuantConv2d(depths[1], depths[2],  stride=(2, 2), kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU())

        self.aspp1 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=1, kernel_size=(3,3), padding=(1,1), bias=False),
                                        qnn.QuantReLU(),)

        self.aspp2 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,2), kernel_size=(3,3), padding=(1,2), bias=False),
                                        qnn.QuantReLU(),)
                                        
        self.aspp3 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,3), kernel_size=(3,3), padding=(1,3), bias=False),
                                        qnn.QuantReLU(),)
        self.aspp4 = torch.nn.Sequential(  qnn.QuantConv2d(depths[2]*2 , depths[3], stride=(1, 1), dilation=(1,4), kernel_size=(3,3), padding=(1,4), bias=False),
                                        qnn.QuantReLU(),)

        self.shrink = torch.nn.Sequential(  qnn.QuantConv2d(depths[3]*4, depths[3], stride=(1, 1), dilation=1, kernel_size=(1,1), padding=(0,0), bias=False),
                                            qnn.QuantReLU(),
        )

        self.dec =torch.nn.Sequential (
                                qnn.QuantConvTranspose2d(depths[3], depths[4], 3, stride=2, padding=1, bias=False, output_padding=1),qnn.QuantReLU(),
                                qnn.QuantConvTranspose2d(depths[4], depths[5], 3, stride=2, padding=1, bias=False, output_padding=1),qnn.QuantReLU(),
                                qnn.QuantConvTranspose2d(depths[5], depths[6],  3, stride=2, padding=1, bias=False, output_padding=1),qnn.QuantReLU()
        )

        self.output = torch.nn.Sequential(  qnn.QuantConv2d(depths[6], 5, stride=(1, 1), kernel_size=(3,3), padding=1, bias=False),
                                            nn.Tanh()
        )         
    def bottleneck(self,FM) :
        fm1= self.aspp1(FM)
        fm2= self.aspp2(FM)
        fm3= self.aspp3(FM)
        fm4= self.aspp4(FM)
        return self.shrink(QuantTensor.cat([fm1,fm2,fm3,fm4],dim=1))

    def forward(self,img0,img1, timestep=0.5):
        
        B,C,H,W= img0.shape
        encoder_FM0 = self.enc(self.quant_input(img0))
        encoder_FM1 = self.enc(self.quant_input(img1))

        encoder_FM =  QuantTensor.cat([encoder_FM0,encoder_FM1],dim=1)
        bottleneck_FM = self.bottleneck(encoder_FM)
        dec_FM = self.dec(bottleneck_FM)
        out =    self.output(dec_FM)
        
        del encoder_FM
        del encoder_FM0
        del encoder_FM1
        del bottleneck_FM
        del dec_FM

        return out

