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

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class SinogramNpyDataset(Dataset):
    """
    用于处理.npy格式正弦图数据的数据集类
    预加载所有数据到内存以提高训练速度，使用float16减少内存占用
    """
    def __init__(self, data_dir, is_train=True, transform=None):
        """
        初始化数据集，预加载所有数据到内存
        
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
            
        print(f"找到 {len(self.complete_files)} 对{'训练' if is_train else '测试'}数据文件")
        print("开始预加载所有数据到内存（使用float16减少内存占用）...")
        
        # 预加载所有数据到内存
        self.complete_sinograms = []
        self.incomplete_sinograms = []
        
        for i in tqdm(range(len(self.complete_files)), desc=f"预加载{'训练' if is_train else '测试'}数据"):
            complete_path = os.path.join(self.base_dir, self.complete_files[i])
            incomplete_path = os.path.join(self.base_dir, self.incomplete_files[i])
            
            # 加载数据
            complete_sinogram = np.load(complete_path)
            incomplete_sinogram = np.load(incomplete_path)
            
            # 转换为PyTorch张量并使用float16减少内存占用
            complete_sinogram = torch.from_numpy(complete_sinogram).half()  # 使用half()而不是float()
            incomplete_sinogram = torch.from_numpy(incomplete_sinogram).half()  # 使用half()而不是float()
            
            # 存储到内存
            self.complete_sinograms.append(complete_sinogram)
            self.incomplete_sinograms.append(incomplete_sinogram)
            
            # 打印第一个样本的形状和数据类型作为参考
            if i == 0:
                print(f"数据形状示例 - 完整: {complete_sinogram.shape}, 不完整: {incomplete_sinogram.shape}")
                print(f"数据类型: {complete_sinogram.dtype}, 内存占用减少约50%")
                    
        
        print(f"成功预加载 {len(self.complete_sinograms)} 对数据到内存（使用float16格式）")
    
    def __len__(self):
        return len(self.complete_sinograms)
    
    def __getitem__(self, idx):
        # 直接从内存获取数据
        complete_sinogram = self.complete_sinograms[idx]
        incomplete_sinogram = self.incomplete_sinograms[idx]
        
        # 转换回float32以匹配模型
        complete_sinogram = complete_sinogram.float()
        incomplete_sinogram = incomplete_sinogram.float()
        
        # 应用变换（如果有）
        if self.transform:
            complete_sinogram = self.transform(complete_sinogram)
            incomplete_sinogram = self.transform(incomplete_sinogram)
        
        return incomplete_sinogram, complete_sinogram


class SparseSinogramDataset(Dataset):
    """
    用于处理.npy格式正弦图数据的数据集类
    强制使用稀疏矩阵存储所有数据以最大程度减少内存占用
    """
    def __init__(self, data_dir, is_train=True, transform=None, return_dense=True):
        """
        初始化数据集，强制使用稀疏矩阵预加载数据到内存
        
        Args:
            data_dir (str): 数据目录
            is_train (bool): 是否为训练集
            transform (callable, optional): 可选的数据转换
            return_dense (bool): 是否在__getitem__中返回密集张量，设为False可与支持稀疏张量的模型一起使用
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        self.return_dense = return_dense
        
        # 确定加载的子目录
        subset_dir = "train" if is_train else "test"
        self.base_dir = os.path.join(data_dir, subset_dir)
        
        # 获取完整和不完整正弦图的文件列表
        self.complete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("complete_")])
        self.incomplete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("incomplete_")])
        
        # 确保文件数量一致
        assert len(self.complete_files) == len(self.incomplete_files), \
            f"完整文件数量({len(self.complete_files)})与不完整文件数量({len(self.incomplete_files)})不匹配"
            
        print(f"找到 {len(self.complete_files)} 对{'训练' if is_train else '测试'}数据文件")
        print(f"开始预加载所有数据到内存（强制使用稀疏存储）...")
        
        # 记录总内存统计
        self.original_size = 0  # 原始数据大小（如果是密集存储）
        self.sparse_size = 0    # 实际使用的稀疏存储大小
        self.total_elements = 0 # 总元素数
        self.nonzero_elements = 0 # 非零元素数
        
        # 预加载所有数据到内存（仅稀疏格式）
        self.complete_sparse = []
        self.incomplete_sparse = []
        self.shapes = []  # 存储原始形状
        
        for i in tqdm(range(len(self.complete_files)), desc=f"预加载{'训练' if is_train else '测试'}数据"):
            complete_path = os.path.join(self.base_dir, self.complete_files[i])
            incomplete_path = os.path.join(self.base_dir, self.incomplete_files[i])
            
            try:
                # 加载数据
                complete_np = np.load(complete_path)
                incomplete_np = np.load(incomplete_path)
                
                # 保存原始形状
                self.shapes.append(complete_np.shape)
                
                # 统计非零元素数量
                complete_nonzeros = np.count_nonzero(complete_np)
                incomplete_nonzeros = np.count_nonzero(incomplete_np)
                
                # 更新统计信息
                self.total_elements += complete_np.size + incomplete_np.size
                self.nonzero_elements += complete_nonzeros + incomplete_nonzeros
                self.original_size += (complete_np.size + incomplete_np.size) * 4  # 假设float32
                
                # 将数据转换为稀疏张量
                complete_sparse = self._to_sparse_tensor(complete_np)
                incomplete_sparse = self._to_sparse_tensor(incomplete_np)
                
                # 估计稀疏存储大小
                complete_sparse_size = self._estimate_sparse_size(complete_sparse)
                incomplete_sparse_size = self._estimate_sparse_size(incomplete_sparse)
                self.sparse_size += complete_sparse_size + incomplete_sparse_size
                
                # 存储稀疏张量
                self.complete_sparse.append(complete_sparse)
                self.incomplete_sparse.append(incomplete_sparse)
                
                # 打印第一个样本的信息作为参考
                if i == 0:
                    print(f"数据形状示例: {complete_np.shape}")
                    print(f"稀疏存储信息:")
                    print(f"  - 完整数据: 非零元素率 {complete_nonzeros/complete_np.size:.2%}")
                    print(f"  - 不完整数据: 非零元素率 {incomplete_nonzeros/incomplete_np.size:.2%}")
                    print(f"  - 索引形状: {complete_sparse.indices().shape}")
                    print(f"  - 值数量: {complete_sparse.values().shape[0]}")
                    
            except Exception as e:
                print(f"加载文件时出错 ({complete_path} 或 {incomplete_path}): {e}")
                # 创建一个空的稀疏张量作为占位符
                shape = (182, 365, 1764) if len(self.shapes) > 0 else (112, 225, 1024)  # 使用之前看到的形状或默认形状
                empty_sparse = torch.sparse_coo_tensor(
                    torch.LongTensor(np.zeros((3, 0))),  # 空索引
                    torch.tensor([], dtype=torch.float16),  # 空值
                    torch.Size(shape)
                )
                self.complete_sparse.append(empty_sparse)
                self.incomplete_sparse.append(empty_sparse)
                self.shapes.append(shape)
        
        # 显示内存统计
        sparsity = 1.0 - (self.nonzero_elements / self.total_elements) if self.total_elements > 0 else 0
        savings = (1.0 - self.sparse_size / self.original_size) * 100 if self.original_size > 0 else 0
        
        print(f"\n数据存储统计:")
        print(f"  - 总元素数: {self.total_elements:,}")
        print(f"  - 非零元素数: {self.nonzero_elements:,} ({(1-sparsity):.2%})")
        print(f"  - 零元素数: {self.total_elements - self.nonzero_elements:,} ({sparsity:.2%})")
        print(f"  - 原始数据大小(估计): {self.original_size/1024/1024:.2f} MB")
        print(f"  - 稀疏存储大小(估计): {self.sparse_size/1024/1024:.2f} MB")
        print(f"  - 内存节省: {savings:.2f}%")
        
        print(f"成功预加载 {len(self.complete_sparse)} 对数据到内存 (稀疏格式)")
    
    def _to_sparse_tensor(self, numpy_array):
        """将NumPy数组转换为PyTorch稀疏张量"""
        # 获取非零元素的索引和值
        indices = np.nonzero(numpy_array)
        values = numpy_array[indices]
        
        # 创建稀疏张量 (使用float16减少内存)
        sparse_tensor = torch.sparse_coo_tensor(
            torch.LongTensor(np.vstack(indices)), 
            torch.tensor(values, dtype=torch.float16),
            torch.Size(numpy_array.shape)
        )
        
        return sparse_tensor
    
    def _estimate_sparse_size(self, sparse_tensor):
        """估计稀疏张量占用的内存大小 (字节)"""
        # 索引: int64 (8字节) * 维度数 * 非零元素数
        indices_size = 8 * sparse_tensor.indices().shape[0] * sparse_tensor.indices().shape[1]
        
        # 值: float16 (2字节) * 非零元素数
        values_size = 2 * sparse_tensor.values().shape[0]
        
        # 返回总大小 (字节)
        return indices_size + values_size
    
    def __len__(self):
        return len(self.complete_sparse)
    
    def __getitem__(self, idx):
        # 获取对应的稀疏张量
        complete_tensor = self.complete_sparse[idx]
        incomplete_tensor = self.incomplete_sparse[idx]
        
        # 如果需要返回密集张量
        if self.return_dense:
            complete_tensor = complete_tensor.to_dense()
            incomplete_tensor = incomplete_tensor.to_dense()
            
            # 应用变换（如果有）
            if self.transform:
                complete_tensor = self.transform(complete_tensor)
                incomplete_tensor = self.transform(incomplete_tensor)
            
            # 转换为float32返回给模型
            return incomplete_tensor.float(), complete_tensor.float()
        else:
            # 直接返回稀疏张量 (某些模型可以直接处理稀疏张量)
            # 注意：大多数模型和损失函数不支持稀疏输入，所以这个选项要谨慎使用
            
            # 由于稀疏张量不支持大多数变换，这里我们跳过transform
            # 如果有变换需求，需要在模型内部处理
            
            # 转换为float32
            return incomplete_tensor.float(), complete_tensor.float()
# 混合精度训练函数
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, amp_enabled=True):
    """
    使用混合精度训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前训练轮次
        amp_enabled: 是否启用混合精度训练
    
    Returns:
        损失和评估指标的平均值
    """
    from torch.cuda.amp import autocast, GradScaler
    
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    # 初始化梯度缩放器，用于混合精度训练
    scaler = GradScaler(enabled=amp_enabled)
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
        for inputs, targets in pbar:
            # 将数据移至设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用混合精度进行前向传播和损失计算
            with autocast(enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 使用梯度缩放器进行反向传播和优化
            if amp_enabled:
                # 缩放损失并反向传播
                scaler.scale(loss).backward()
                # 缩放优化器的步骤
                scaler.step(optimizer)
                # 更新缩放器
                scaler.update()
            else:
                # 常规的反向传播和优化
                loss.backward()
                optimizer.step()
            
            # 更新统计信息
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            
            # 计算指标 (在no_grad上下文中使用FP32精度)
            with torch.no_grad():
                # 对于指标计算，我们使用CPU并将数据转换为float32以确保准确性
                outputs_for_metrics = outputs.detach().cpu()
                targets_for_metrics = targets.cpu()
                
                psnr_val = calculate_psnr(outputs_for_metrics, targets_for_metrics)
                ssim_val = calculate_ssim(outputs_for_metrics, targets_for_metrics)
                
                psnr_meter.update(psnr_val, batch_size)
                ssim_meter.update(ssim_val, batch_size)
            
            # 释放内存
            if amp_enabled:
                # 清除缓存，释放显存
                torch.cuda.empty_cache()
            
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
    train_dataset = SparseSinogramDataset(args.data_dir, is_train=True)
    # test_dataset = SinogramNpyDataset(args.data_dir, is_train=False)
    
    # 从训练集中分离验证集
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    # print(f"Test set size: {len(test_dataset)}")
    
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
    
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
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
            model, train_loader, criterion, optimizer, device, epoch, amp_enabled=True
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
                model, val_dataset, device, 
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
    # print("Evaluating on test set...")
    # test_loss, test_psnr, test_ssim = validate(
    #     model, test_loader, criterion, device
    # )
    # print(f"Test Loss: {test_loss:.4f}, PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")
    
    # 保存最终训练曲线
    plot_training_curves(
        train_losses, val_losses, train_metrics, val_metrics,
        os.path.join(output_dir, 'final_training_curves.png')
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()