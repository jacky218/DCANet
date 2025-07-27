import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functions.dcorrelation import DeformableCorrelationFunction

class DeformableCorrelation(nn.Module):
    """
    单尺度可形变相关性模块
    """
    def __init__(self):
        super(DeformableCorrelation, self).__init__()

    def forward(self, feat1, feat2, offset):
        """
        Args:
            feat1 (torch.Tensor): 第一个特征图 (B, C, H, W)
            feat2 (torch.Tensor): 第二个特征图 (B, C, H, W)
            offset (torch.Tensor): 偏移场 (B, 2, H, W)
        Returns:
            torch.Tensor: 相关性图 (B, H, W)
        """
        return DeformableCorrelationFunction.apply(feat1, feat2, offset)

class MultiScaleDeformableCorrelation(nn.Module):
    """
    多尺度可形变相关性模块
    """
    def __init__(self, scales=[0, 1, 2]):
        """
        Args:
            scales (list): 要进行计算的下采样尺度列表。0代表原分辨率。
        """
        super(MultiScaleDeformableCorrelation, self).__init__()
        self.scales = scales
        self.single_correlation = DeformableCorrelation()

    def forward(self, feat1, feat2, offsets):
        """
        Args:
            feat1 (torch.Tensor): 第一个特征图 (B, C, H, W)
            feat2 (torch.Tensor): 第二个特征图 (B, C, H, W)
            offsets (torch.Tensor): 多尺度偏移场 (B, S*2, H, W), S是尺度数量
        
        Returns:
            torch.Tensor: 聚合后的多尺度相关性图 (B, S, H, W)
        """
        if len(self.scales) != offsets.size(1) // 2:
            raise ValueError("Number of scales must match the offset's channel dimension / 2.")

        all_correlations = []
        
        for i, scale in enumerate(self.scales):
            offset_scale = offsets[:, i*2:(i+1)*2, :, :]
            
            if scale > 0:
                h, w = feat1.shape[2:]
                h_scale, w_scale = h // (2**scale), w // (2**scale)
                
                feat1_scale = F.interpolate(feat1, size=(h_scale, w_scale), mode='bilinear', align_corners=False)
                feat2_scale = F.interpolate(feat2, size=(h_scale, w_scale), mode='bilinear', align_corners=False)
                offset_scale = F.interpolate(offset_scale, size=(h_scale, w_scale), mode='bilinear', align_corners=False) / (2**scale)
            else:
                feat1_scale, feat2_scale = feat1, feat2

            corr = self.single_correlation(feat1_scale, feat2_scale, offset_scale) # (B, H_s, W_s)
            
            if scale > 0:
                corr = F.interpolate(corr.unsqueeze(1), size=feat1.shape[2:], mode='bilinear', align_corners=False).squeeze(1)

            all_correlations.append(corr)
            
        multi_scale_corr = torch.stack(all_correlations, dim=1) # (B, S, H, W)
        
        return multi_scale_corr
