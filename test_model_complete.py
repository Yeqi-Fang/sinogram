"""
简单的端到端测试脚本，用于测试模型对完整数据的处理能力
"""
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from models.sinogram_transformer import SinogramTransformer

def test_model_with_complete_data(sinogram_path, save_dir="./test_results"):
    """
    使用完整数据测试模型，无需数据掩码或增强
    
    Args:
        sinogram_path: 完整正弦图数据的路径
        save_dir: 保存测试结果的目录
    """
    print(f"加载数据: {sinogram_path}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    sinogram = torch.load(sinogram_path)
    print(f"数据形状: {sinogram.shape}")
    
    # 从数据中获取一个小块用于测试
    # 选择中间的切片以确保有足够的信号
    height, width, depth = sinogram.shape
    block_size = min(32, depth)
    mid_idx = depth // 2 - block_size // 2
    test_block = sinogram[:, :, mid_idx:mid_idx+block_size].clone()
    
    if test_block.shape[2] < block_size:
        padded_block = torch.zeros((height, width, block_size))
        padded_block[:, :, :test_block.shape[2]] = test_block
        test_block = padded_block
    
    print(f"测试数据块形状: {test_block.shape}")
    
    # 可视化输入数据
    plt.figure(figsize=(10, 8))
    plt.imshow(test_block[:, :, 0].numpy(), cmap='hot', aspect='auto')
    plt.title("测试输入数据 (第一个切片)")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "test_input.png"))
    plt.close()
    
    # 初始化模型
    print("初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SinogramTransformer(
        sinogram_shape=(height, width, depth),
        block_channels=block_size,
        embed_dim=64,
        num_heads=4,
        pool_size=(4, 4)
    ).to(device)
    
    # 添加批次维度并移至设备
    input_tensor = test_block.unsqueeze(0).to(device)  # [1, H, W, C]
    
    # 前向传播
    print(f"在 {device} 上运行模型...")
    with torch.no_grad():
        output = model(input_tensor)
    
    # 提取输出
    output = output.cpu().squeeze(0)  # [H, W, C]
    print(f"输出形状: {output.shape}")
    
    # 可视化输出
    plt.figure(figsize=(10, 8))
    plt.imshow(output[:, :, 0].numpy(), cmap='hot', aspect='auto')
    plt.title("模型输出 (第一个切片)")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "test_output.png"))
    plt.close()
    
    # 计算输入和输出之间的差异
    diff = torch.abs(output - test_block)
    print(f"最大差异: {diff.max().item():.6f}")
    print(f"平均差异: {diff.mean().item():.6f}")
    
    # 可视化差异
    plt.figure(figsize=(10, 8))
    plt.imshow(diff[:, :, 0].numpy(), cmap='hot', aspect='auto')
    plt.title("输入与输出差异 (第一个切片)")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "test_diff.png"))
    plt.close()
    
    # 可视化多个切片
    num_slices = min(4, block_size)
    fig, axs = plt.subplots(3, num_slices, figsize=(4*num_slices, 12))
    
    for i in range(num_slices):
        slice_idx = i * (block_size // num_slices)
        
        # 输入
        axs[0, i].imshow(test_block[:, :, slice_idx].numpy(), cmap='hot', aspect='auto')
        axs[0, i].set_title(f"输入 (切片 {slice_idx})")
        
        # 输出
        axs[1, i].imshow(output[:, :, slice_idx].numpy(), cmap='hot', aspect='auto')
        axs[1, i].set_title(f"输出 (切片 {slice_idx})")
        
        # 差异
        im = axs[2, i].imshow(diff[:, :, slice_idx].numpy(), cmap='hot', aspect='auto')
        axs[2, i].set_title(f"差异 (切片 {slice_idx})")
    
    plt.colorbar(im, ax=axs.ravel().tolist())
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_multi_slice.png"))
    plt.close()
    
    print(f"测试完成。结果保存在: {save_dir}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用完整数据测试SinogramTransformer模型")
    parser.add_argument("--data_path", type=str, default="./data/synthetic/complete_sinogram.pt",
                      help="完整正弦图数据的路径")
    parser.add_argument("--save_dir", type=str, default="./test_results",
                      help="保存测试结果的目录")
    
    args = parser.parse_args()
    
    test_model_with_complete_data(args.data_path, args.save_dir)