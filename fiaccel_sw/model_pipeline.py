import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
from datamodules.video_data_api import  VideoData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

class FIACCELPipeline(nn.Module):
    """
    Glue together encoder transform, entropy model and decoder transform
    """
    def __init__(
        self,
        flow_estimation:nn.Module,
        ) :
        super().__init__()

        self.flow_estimation = flow_estimation

    
    def meshgrid(self,height, width):
        x_t = torch.matmul(
            torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
        y_t = torch.matmul(
            torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))
        grid_x = x_t.view(1, height, width)
        grid_y = y_t.view(1, height, width)
        return grid_x, grid_y

    @torch.no_grad()
    def inference(self,  video:VideoData, timestep = 0.5):

        imgs=video.video_tensor
        B, T, C, H, W = imgs.shape
        img0,gt,img1 = imgs[:, 0,...],imgs[:,1,...],imgs[:, 2,...]
        out = self.flow_estimation(img0,img1)
        
        mode= 'uniflow'
        
        if mode == 'uniflow' :
            flow = out[:,0:2]
            mask = out[:,2:3]
        elif mode == 'biflow' :
            flow = out[:,0:4]
            mask = out[:,4:5]

        flow = 0.5 * flow
        mask = 0.5 * (1.0 + mask)

        if mode == 'uniflow' :
            grid_x ,grid_y = self.meshgrid(H,W)
            grid_x = torch.autograd.Variable(
            grid_x.repeat([B, 1, 1])).cuda()
            grid_y = torch.autograd.Variable(
            grid_y.repeat([B, 1, 1])).cuda()
            coor_x_1 = grid_x - flow[:, 0, :, :]
            coor_y_1 = grid_y - flow[:, 1, :, :]
            coor_x_2 = grid_x + flow[:, 0, :, :]
            coor_y_2 = grid_y + flow[:, 1, :, :]

            warped_img0 = torch.nn.functional.grid_sample(
            img0,
                torch.stack([coor_x_1,coor_y_1],dim=3),
                padding_mode='border',align_corners=True)
            warped_img1 = torch.nn.functional.grid_sample(
                img1,
                torch.stack([coor_x_2,coor_y_2],dim=3),
                padding_mode='border',align_corners=True)

        elif mode =='biflow':
            mask=torch.sigmoid(mask)
            warped_img0 =  warp(img0, flow[:,  :2])
            warped_img1 =  warp(img1, flow[:, 2:4])

        merged=(warped_img0 * mask + warped_img1 * (1 - mask))

        pred = torch.clamp( merged, 0, 1)

        return pred



    def forward(self, video:VideoData):

        imgs=video.video_tensor
        B, T, C, H, W = imgs.shape
        img0,gt,img1 = imgs[:, 0,...],imgs[:,1,...],imgs[:, 2,...]
        out = self.flow_estimation(img0,img1)
        
        mode= 'uniflow'
     
        if mode == 'uniflow' :
            flow = out[:,0:2]
            mask = out[:,2:3]

            flow = 0.5 * flow
            mask = 0.5 * (1.0 + mask)
            
            grid_x ,grid_y = self.meshgrid(H,W)
            grid_x = torch.autograd.Variable(
            grid_x.repeat([B, 1, 1])).cuda()
            grid_y = torch.autograd.Variable(
            grid_y.repeat([B, 1, 1])).cuda()
            coor_x_1 = grid_x - flow[:, 0, :, :]
            coor_y_1 = grid_y - flow[:, 1, :, :]
            coor_x_2 = grid_x + flow[:, 0, :, :]
            coor_y_2 = grid_y + flow[:, 1, :, :]

            warped_img0 = torch.nn.functional.grid_sample(
            img0,
                torch.stack([coor_x_1,coor_y_1],dim=3),
                padding_mode='border',align_corners=True)
            warped_img1 = torch.nn.functional.grid_sample(
                img1,
                torch.stack([coor_x_2,coor_y_2],dim=3),
                padding_mode='border',align_corners=True)

        elif mode =='biflow':
            flow = out[:,0:4]
            mask = out[:,4:5]
            mask=torch.sigmoid(mask)
            warped_img0 =  warp(img0, flow[:,  :2])
            warped_img1 =  warp(img1, flow[:, 2:4])

        merged=(warped_img0 * mask + warped_img1 * (1 - mask))

        pred = torch.clamp( merged, 0, 1)

        return pred
