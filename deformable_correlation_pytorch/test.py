import torch
from dcorrelation import MultiScaleDeformableCorrelation

def test_multiscale_dcorrelation():
    print("🚀 Testing MultiScaleDeformableCorrelation...")

    batch_size = 244
    channels = 16
    height, width = 20, 30
    scales = [0, 1]  
    num_scales = len(scales)

    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return
    device = torch.device("cuda")
    print(f"Running on device: {device}")

    model = MultiScaleDeformableCorrelation(scales=scales).to(device)

    feat1 = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)
    feat2 = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)
    offset_net_output = torch.randn(batch_size, num_scales * 2, height, width, device=device)
    offsets = torch.tanh(offset_net_output) * 5  # 限制偏移量在 [-5, 5] 像素
    offsets.requires_grad = True

    print("\n--- Input Shapes ---")
    print(f"feat1: {feat1.shape}")
    print(f"feat2: {feat2.shape}")
    print(f"offsets: {offsets.shape}")

    print()
    try:
        output = model(feat1, feat2, offsets)
        print("\n✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        # 期望输出: (B, num_scales, H, W) -> (2, 2, 20, 30)
        assert output.shape == (batch_size, num_scales, height, width)
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # --- 后向传播 ---
    try:
        # 创建一个虚拟的 loss 并反向传播
        loss = output.sum()
        loss.backward()
        print("\n✅ Backward pass successful!")
        
        # 检查梯度是否存在
        assert feat1.grad is not None, "Gradient for feat1 is missing!"
        assert feat2.grad is not None, "Gradient for feat2 is missing!"
        assert offsets.grad is not None, "Gradient for offsets is missing!"

        print("Gradient shapes:")
        print(f"feat1.grad: {feat1.grad.shape}")
        print(f"feat2.grad: {feat2.grad.shape}")
        print(f"offsets.grad: {offsets.grad.shape}")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return

    print("\n🎉 Test completed successfully! 🎉")


if __name__ == '__main__':
    test_multiscale_dcorrelation()

