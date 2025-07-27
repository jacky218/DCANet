import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure this import works based on your project structure
# This is the custom CUDA operator we created previously.
from dcorrelation import MultiScaleDeformableCorrelation

# This Correlation Filter is based on your 'Corr_Attn' class.
# It's used to filter the correlation maps produced by the custom operator.
class CorrelationMapFilter(nn.Module):
    """
    Filters a correlation map by applying channel-wise attention.
    The 'channels' of the correlation map correspond to the different scales.
    """
    def __init__(self, num_scales, k_size=3):
        super(CorrelationMapFilter, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # Takes mean and max stats across scales
        self.conv1 = nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # Takes variance stat across scales
        self.conv2 = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, corr_map):
        """
        Args:
            corr_map (torch.Tensor): Correlation map of shape (B, S, H, W),
                                     where S is the number of scales.
        """
        avg_out = torch.mean(corr_map, dim=1, keepdim=True)
        max_out, _ = torch.max(corr_map, dim=1, keepdim=True)
        var_out = torch.var(corr_map, dim=1, keepdim=True)

        y1 = torch.cat([avg_out, max_out], dim=1)
        
        # Calculate attention map based on correlation statistics
        attention = self.sigmoid(self.conv1(y1) + self.conv2(var_out))

        # The attention map is (B, 1, H, W). We apply it to the multi-scale
        # correlation map (B, S, H, W) using broadcasting.
        return corr_map * attention

# Helper functions from your code
def create_grid_like(t):
    h, w, device = *t.shape[-2:], t.device
    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device=device, dtype=t.dtype),
        torch.arange(h, device=device, dtype=t.dtype),
        indexing='xy'), dim=0)
    grid.requires_grad = False
    return grid

def normalize_grid(grid):
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid[1], grid[0] # PyTorch meshgrid's y, x ordering
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0
    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    return torch.stack((grid_w, grid_h), dim=0)


class MS_DefCorrelation_Module(nn.Module):
    """
    This class replaces the inefficient `einsum` based correlation from your
    `Get_DefCorrelation` with the efficient `MultiScaleDeformableCorrelation` operator.
    """
    def __init__(self, channels, scales=(0, 1, 2)):
        super().__init__()
        num_scales = len(scales)
        reduction_channel = channels // 16 if channels > 16 else 1

        # 1. Multi-Scale Deformable Correlation Sub-module
        self.ms_dcorrelation = MultiScaleDeformableCorrelation(scales=scales)

        # 2. Offset Prediction Networks
        # These networks predict offsets for the multi-scale correlation.
        # The output channels must be num_scales * 2 (x and y for each scale).
        self.ofs_encoder_l = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_scales * 2, 3, padding=1)
        )
        self.ofs_encoder_r = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_scales * 2, 3, padding=1)
        )

        # 3. Correlation Map Filter
        # This filters the (B, S, H, W) output from the custom operator.
        self.corr_filter = CorrelationMapFilter(num_scales=num_scales)
        
        # 4. Long-term Temporal Enhanced Sub-module (identical to your implementation)
        self.down_conv_agg = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3), padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3), padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3), padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.agg_weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.conv_back_agg = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

        # 5. Feature Fusion Weights
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # --- Part 1: Long-term Temporal Enhancement ---
        # This part is identical to your original code.
        x_agg = self.down_conv_agg(x)
        aggregated_x = (self.spatial_aggregation1(x_agg) * self.agg_weights[0] +
                        self.spatial_aggregation2(x_agg) * self.agg_weights[1] +
                        self.spatial_aggregation3(x_agg) * self.agg_weights[2])
        long_term_map = self.sigmoid(self.conv_back_agg(aggregated_x)) - 0.5

        # --- Part 2: Multi-Scale Deformable Correlation ---
        # Get neighboring frames (t+1 and t-1)
        # To handle boundaries, we repeat the first and last frames.
        left_move = torch.cat([x[:, :, 1:], x[:, :, -1:]], 2)
        right_move = torch.cat([x[:, :, :1], x[:, :, :-1]], 2)

        # Reshape for 2D convolution: (B, C, T, H, W) -> (B*T, C, H, W)
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        left_move_flat = left_move.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        right_move_flat = right_move.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)

        # Predict offsets using the current frame and its neighbor
        offset_lm = self.ofs_encoder_l(torch.cat([x_flat, left_move_flat], dim=1))
        offset_rm = self.ofs_encoder_r(torch.cat([x_flat, right_move_flat], dim=1))
        
        # === EFFICIENT CORRELATION CALCULATION ===
        # Call the custom CUDA operator. This replaces the expensive einsum.
        # The operator performs sampling and correlation internally.
        corr_lm = self.ms_dcorrelation(x_flat, left_move_flat, offset_lm) # (B*T, S, H, W)
        corr_rm = self.ms_dcorrelation(x_flat, right_move_flat, offset_rm) # (B*T, S, H, W)

        # Filter the resulting correlation maps
        filtered_corr_lm = self.corr_filter(corr_lm)
        filtered_corr_rm = self.corr_filter(corr_rm)

        # To modulate the features, we need to explicitly warp the neighbors
        # using the same offsets. We first need to normalize the offsets for grid_sample.
        # We take the offsets for the highest resolution scale (scale 0) for warping.
        offset_lm_scale0 = offset_lm[:, :2, :, :]
        offset_rm_scale0 = offset_rm[:, :2, :, :]
        
        grid = create_grid_like(x_flat)
        vgrid_lm = normalize_grid(grid) + (offset_lm_scale0 / torch.tensor([W-1, H-1], device=x.device).view(1,2,1,1))
        vgrid_rm = normalize_grid(grid) + (offset_rm_scale0 / torch.tensor([W-1, H-1], device=x.device).view(1,2,1,1))

        warped_lm = F.grid_sample(left_move_flat, vgrid_lm.permute(0,2,3,1), mode='bilinear', padding_mode='zeros', align_corners=False)
        warped_rm = F.grid_sample(right_move_flat, vgrid_rm.permute(0,2,3,1), mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Use the mean of the filtered correlation scales as an attention map
        attn_lm = torch.mean(filtered_corr_lm, dim=1, keepdim=True)
        attn_rm = torch.mean(filtered_corr_rm, dim=1, keepdim=True)

        # Calculate the short-term features from correlation
        features_lm = warped_lm * (self.sigmoid(attn_lm) - 0.5)
        features_rm = warped_rm * (self.sigmoid(attn_rm) - 0.5)

        short_term_features = self.fusion_weights[0] * features_lm + self.fusion_weights[1] * features_rm
        
        # Reshape back to 5D: (B*T, C, H, W) -> (B, C, T, H, W)
        short_term_features = short_term_features.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        # --- Part 3:  Fusion ---
        # Combine the short-term correlation features with the long-term map
        return short_term_features * long_term_map


if __name__ == "__main__":
    # Define test parameters
    batch_size = 2
    channels = 64
    time_steps = 10
    height, width = 28, 28
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Running test on {device}")
    
    # Create a dummy input tensor
    inp = torch.randn((batch_size, channels, time_steps, height, width)).to(device)

    # Instantiate the new module with 3 scales (original, 1/2, 1/4)
    # The custom op will handle the downsampling automatically.
    try:
        corr_module = MS_DefCorrelation_Module(channels=channels, scales=(0, 1, 2)).to(device)

        # Perform a forward pass
        print(f"\nInput shape: {inp.shape}")
        output = corr_module(inp)
        print("✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")

        # Check that output shape is correct
        assert output.shape == inp.shape

        # Perform a backward pass to check gradients
        loss = output.sum()
        loss.backward()
        print("✅ Backward pass successful!")
        
        a_param = next(corr_module.parameters())
        assert a_param.grad is not None
        print("✅ Gradients computed successfully.")

        print("\n🎉 Test completed successfully! 🎉")

    except ImportError as e:
        print("\n❌ IMPORT ERROR: Could not import the 'dcorrelation' module.")
        print("   Please ensure the custom operator is compiled and in your Python path.")
    except Exception as e:
        import traceback
        print(f"\n❌ An error occurred during the test: {e}")
        traceback.print_exc()

