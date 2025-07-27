import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import numpy
import torch

class Corr_Attn(nn.Module):

    def __init__(self, k_size=3):
        super(Corr_Attn, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.conv2 = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_= torch.max(x,dim=1,keepdim=True)
        var_out = torch.var(x,dim=1,keepdim=True)

        y1 = torch.cat([avg_out,max_out],dim=1)

        y = self.sigmoid(self.conv1(y1)+self.conv2(var_out))

        return x * y

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

class Get_DefCorrelation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels // 16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

        self.ofs_encoder_l = nn.Conv2d(channels, 2, kernel_size=3, padding=1, stride=2)
        self.ofs_encoder_r = nn.Conv2d(channels, 2, kernel_size=3, padding=1, stride=2)

        self.corrAttn = Corr_Attn()

        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: 2, 128, 244, 28, 28 -> bcthw
        # x2: 2, 128, 244, 28, 28 -> bcthw
        #:affinities: 2,244,28,28,28,28 ->bthwsd
        #:affinities2: 2,244,28,28,28,28 ->bthwsd

        x2 = self.down_conv2(x)

        left_move = torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2)
        right_move = torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2)

        ofs_lm = x+left_move
        B,C,T,H,W = ofs_lm.shape

        ofs_lm = ofs_lm.permute(0,2,1,3,4).contiguous().view(B*T,C,H,W)
        offset_lm = self.ofs_encoder_l(ofs_lm)

        _,_,H2,W2 = offset_lm.shape

        ofs_rm = x+right_move
        ofs_rm = ofs_rm.permute(0,2,1,3,4).contiguous().view(B*T,C,H,W)
        offset_rm = self.ofs_encoder_r(ofs_rm)

        grid = create_grid_like(offset_lm)*2

        vgrid_lm = grid + offset_lm

        vgrid_rm = grid + offset_rm

        vgrid_lm_scaled = normalize_grid(vgrid_lm)
        vgrid_rm_scaled = normalize_grid(vgrid_rm)
        
        left_part = left_move.permute(0,2,1,3,4).contiguous().view(B*T,C,H,W)
        right_part = right_move.permute(0,2,1,3,4).contiguous().view(B*T,C,H,W)

        left_part = F.grid_sample(
            left_part,
            vgrid_lm_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        right_part = F.grid_sample(
            right_part,
            vgrid_rm_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        # x:torch.Size([2, 128, 244, 28, 28])

        left_part = left_part.view(B, T, C, H2, W2).contiguous().permute(0,2,1,3,4)
        right_part = right_part.view(B, T, C, H2, W2).contiguous().permute(0,2,1,3,4)

        affinities1 = torch.einsum('bcthw,bctsd->bthwsd', x, left_part)
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, right_part)

        # N, T, H, W, H1, W1 = affinities1.shape
        # MC1 = affinities1.view(T, H * W, H1 * W1)
        # MC2 = affinities2.view(T, H * W, H1 * W1)
        # MC1 = MC1.detach().cpu().numpy()
        # MC2 = MC2.detach().cpu().numpy()
        # numpy.save("left.npy", MC1)
        # numpy.save("right.npy", MC2)

        affinities1 = affinities1.view(B*T,H,W,(H2)*(W2)).contiguous().permute(0,3,1,2)
        affinities2 = affinities2.view(B*T,H,W,(H2)*(W2)).contiguous().permute(0,3,1,2)



        affinities1 = self.corrAttn(affinities1)
        affinities2 = self.corrAttn(affinities2)

        # torch.save(affinities1, "/home/honsen/tartan/TFNet-main/left.pth")
        # torch.save(affinities2, "/home/honsen/tartan/TFNet-main/right.pth")

        affinities1 = affinities1.view(B,T,H2,W2,H,W).contiguous().permute(0,1,4,5,2,3)
        affinities2 = affinities2.view(B,T,H2,W2,H,W).contiguous().permute(0,1,4,5,2,3)



        features1 = torch.einsum('bctsd,bthwsd->bcthw', left_part,
                                self.sigmoid(affinities1) - 0.5) * self.weights2[0] 
        features2 = torch.einsum('bctsd,bthwsd->bcthw', right_part,
                                self.sigmoid(affinities2) - 0.5) * self.weights2[1]
        features = features1+features2

        x = self.down_conv(x)

        aggregated_x = self.spatial_aggregation1(x) * self.weights[0] + self.spatial_aggregation2(x) * self.weights[1] \
                       + self.spatial_aggregation3(x) * self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        # torch.save(F.sigmoid(aggregated_x) - 0.5, "/home/honsen/tartan/TFNet-main/weight_map.pth")

        return features * (self.sigmoid(aggregated_x) - 0.5)

class Get_Correlation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels // 16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: 2, 128, 244, 28, 28 -> bcthw
        # x2: 2, 128, 244, 28, 28 -> bcthw
        #:affinities: 2,244,28,28,28,28 ->bthwsd
        #:affinities2: 2,244,28,28,28,28 ->bthwsd

        x2 = self.down_conv2(x)
        temp = torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2)
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x,
                                  torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2))  # repeat the last frame
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x,
                                   torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2))  # repeat the first frame
        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2),
                                self.sigmoid(affinities) - 0.5) * self.weights2[0] + \
                   torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2),
                                self.sigmoid(affinities2) - 0.5) * self.weights2[1]

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x) * self.weights[0] + self.spatial_aggregation2(x) * self.weights[1] \
                       + self.spatial_aggregation3(x) * self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        return features * (self.sigmoid(aggregated_x) - 0.5)

if __name__ == "__main__":

    inp = torch.randn((2, 128, 244, 28, 28)).cuda()# batchsize, C, T, 28, 28
    corr = Get_DefCorrelation(128).cuda()
    output = corr(inp)
    print()
