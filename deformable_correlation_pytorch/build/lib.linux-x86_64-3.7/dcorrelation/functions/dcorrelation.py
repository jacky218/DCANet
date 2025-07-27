import torch
from torch.autograd import Function
from .. import _C

class DeformableCorrelationFunction(Function):
    @staticmethod
    def forward(ctx, feat1, feat2, offset):
        """
        前向传播.
        Args:
            feat1 (torch.Tensor): 第一个特征图 (B, C, H, W)
            feat2 (torch.Tensor): 第二个特征图 (B, C, H, W)
            offset (torch.Tensor): 偏移场 (B, 2, H, W), channel 0 for x, 1 for y
        """
        ctx.save_for_backward(feat1, feat2, offset)
        
        output = _C.forward(feat1, feat2, offset)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        后向传播.
        """
        feat1, feat2, offset = ctx.saved_tensors
        
        grad_feat1, grad_feat2, grad_offset = _C.backward(grad_output.contiguous(), feat1, feat2, offset)
        
        return grad_feat1, grad_feat2, grad_offset, None # None for non-tensor inputs
