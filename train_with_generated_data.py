"""
使用生成的数据训练SinogramTransformer模型。
该脚本加载由generate_specific_data.py生成的数据集。
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from models.sinogram_transformer import SinogramTransformer
from utils.metrics import calculate_psnr, calculate_ssim, calculate_metrics, AverageMeter

# 数据集类定义
class SinogramNpyDataset(Dataset):
    """
    用于处理.npy格式正弦图数据的数据集类
    """
    def __init__(self, data_dir, is_train=True, transform=None):
        """
        初始化数据集
        
        Args:
            data_dir (str): 数据目录
            is_train (bool): 是否为训练集
            transform (callable, optional): 可选的数据转换
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        
        # 确定加载的子目录
        subset_dir = "train" if is_train else "test"
        self.base_dir = os.path.join(data_dir, subset_dir)
        
        # 获取完整和不完整正弦图的文件列表
        self.complete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("complete_")])
        self.incomplete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("incomplete_")])
        
        # 确保文件数量一致
        assert len(self.complete_files) == len(self.incomplete_files), \
            f"完整文件数量({len(self.complete_files)})与不完整文件数量({len(self.incomplete_files)})不匹配"
    
    def __len__(self):
        return len(self.complete_files)
    
    def __getitem__(self, idx):
        # 加载完整和不完整的正弦图
        complete_path = os.path.join(self.base_dir, self.complete_files[idx])
        incomplete_path = os.path.join(self.base_dir, self.incomplete_files[idx])
        
        complete_sinogram = np.load(complete_path)
        incomplete_sinogram = np.load(incomplete_path)
        
        # 转换为PyTorch张量
        complete_sinogram = torch.from_numpy(complete_sinogram).float()
        incomplete_sinogram = torch.from_numpy(incomplete_sinogram).float()
        
        # 应用变换（如果有）
        if self.transform:
            complete_sinogram = self.transform(complete_sinogram)
            incomplete_sinogram = self.transform(incomplete_sinogram)
        
        return incomplete_sinogram, complete_sinogram

# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
        for inputs, targets in pbar:
            # 将数据移至设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            
            # 计算指标
            with torch.no_grad():
                psnr_val = calculate_psnr(outputs.detach(), targets)
                ssim_val = calculate_ssim(outputs.detach(), targets)
                psnr_meter.update(psnr_val, batch_size)
                ssim_meter.update(ssim_val, batch_size)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'psnr': f"{psnr_meter.avg:.2f}",
                'ssim': f"{ssim_meter.avg:.4f}"
            })
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg

# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for inputs, targets in pbar:
                # 将数据移至设备
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 更新统计信息
                batch_size = inputs.size(0)
                losses.update(loss.item(), batch_size)
                
                # 计算指标
                metrics = calculate_metrics(outputs, targets)
                psnr_meter.update(metrics['psnr'], batch_size)
                ssim_meter.update(metrics['ssim'], batch_size)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{losses.avg:.4f}",
                    'psnr': f"{psnr_meter.avg:.2f}",
                    'ssim': f"{ssim_meter.avg:.4f}"
                })
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg

# 保存示例预测结果
def save_prediction_sample(model, dataset, device, output_dir, epoch, num_samples=2):
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            # 获取样本
            incomplete, complete = dataset[i]
            
            # 添加批次维度并移至设备
            incomplete = incomplete.unsqueeze(0).to(device)
            complete = complete.unsqueeze(0).to(device)
            
            # 模型预测
            prediction = model(incomplete)
            
            # 移回CPU
            incomplete = incomplete.cpu().squeeze(0).numpy()
            complete = complete.cpu().squeeze(0).numpy()
            prediction = prediction.cpu().squeeze(0).numpy()
            
            # 选择几个切片用于可视化
            depth = incomplete.shape[2]
            slice_indices = [0, depth//4, depth//2, 3*depth//4, depth-1]
            
            for slice_idx in slice_indices:
                plt.figure(figsize=(15, 5))
                
                # 可视化不完整输入
                plt.subplot(1, 3, 1)
                plt.imshow(incomplete[:, :, slice_idx], cmap='hot', aspect='auto')
                plt.title(f"Incomplete (Slice {slice_idx})")
                plt.colorbar()
                
                # 可视化预测
                plt.subplot(1, 3, 2)
                plt.imshow(prediction[:, :, slice_idx], cmap='hot', aspect='auto')
                plt.title(f"Prediction (Slice {slice_idx})")
                plt.colorbar()
                
                # 可视化完整目标
                plt.subplot(1, 3, 3)
                plt.imshow(complete[:, :, slice_idx], cmap='hot', aspect='auto')
                plt.title(f"Complete (Slice {slice_idx})")
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{i+1}_slice_{slice_idx}_epoch_{epoch+1}.png"))
                plt.close()

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path):
    """
    绘制训练曲线
    """
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # PSNR曲线
    plt.subplot(2, 2, 2)
    plt.plot([metrics['psnr'] for metrics in train_metrics], label='Train PSNR')
    plt.plot([metrics['psnr'] for metrics in val_metrics], label='Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR')
    plt.legend()
    plt.grid(True)
    
    # SSIM曲线
    plt.subplot(2, 2, 3)
    plt.plot([metrics['ssim'] for metrics in train_metrics], label='Train SSIM')
    plt.plot([metrics['ssim'] for metrics in val_metrics], label='Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train SinogramTransformer with generated data")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data/specific",
                        help="包含训练和测试数据的目录")
    
    # 模型参数
    parser.add_argument("--height", type=int, default=112,
                        help="正弦图高度（角度采样数）")
    parser.add_argument("--width", type=int, default=225,
                        help="正弦图宽度（径向采样数）")
    parser.add_argument("--depth", type=int, default=1024,
                        help="正弦图深度（通道数）")
    parser.add_argument("--block_size", type=int, default=32,
                        help="块大小")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Transformer头数")
    parser.add_argument("--pool_size", type=int, default=(4, 4),
                        help="池化大小")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例（从训练集中分离）")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="保存模型的轮数间隔")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据集
    train_dataset = SinogramNpyDataset(args.data_dir, is_train=True)
    test_dataset = SinogramNpyDataset(args.data_dir, is_train=False)
    
    # 从训练集中分离验证集
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = SinogramTransformer(
        sinogram_shape=(args.height, args.width, args.depth),
        block_channels=args.block_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        pool_size=args.pool_size
    ).to(device)
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练和验证历史
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_val_loss = float('inf')
    
    # 训练循环
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        # 训练一个轮次
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        train_metrics.append({'psnr': train_psnr, 'ssim': train_ssim})
        
        # 验证
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_metrics.append({'psnr': val_psnr, 'ssim': val_ssim})
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        print(f"Epoch {epoch+1}/{args.num_epochs} - "
              f"Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f} - "
              f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f} - "
              f"LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        
        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # 保存样本预测
            save_prediction_sample(
                model, test_dataset, device, 
                os.path.join(sample_dir, f'epoch_{epoch+1}'),
                epoch
            )
            
            # 绘制训练曲线
            plot_training_curves(
                train_losses, val_losses, train_metrics, val_metrics,
                os.path.join(output_dir, 'training_curves.png')
            )
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # 保存最终模型
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_losses[-1],
        'val_psnr': val_metrics[-1]['psnr'],
        'val_ssim': val_metrics[-1]['ssim']
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # 在测试集上评估
    print("Evaluating on test set...")
    test_loss, test_psnr, test_ssim = validate(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}, PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")
    
    # 保存最终训练曲线
    plot_training_curves(
        train_losses, val_losses, train_metrics, val_metrics,
        os.path.join(output_dir, 'final_training_curves.png')
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()