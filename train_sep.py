"""
BlockwiseSinogramTransformer 

该模型将大型sinogram (182, 365, 1764)拆分成42个块，每个块(182, 365, 42)，
并使用UNet+Transformer的组合架构进行处理。
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import math

# ---------------------------
# 数据集类
# ---------------------------

class BlockwiseSinogramDataset(Dataset):
    """
    将正弦图数据分块处理的数据集类
    每个(182, 365, 1764)的sinogram被拆分成42个(182, 365, 42)的块
    """
    def __init__(self, data_dir, is_train=True, transform=None, blocks_count=42):
        """
        初始化数据集
        
        Args:
            data_dir (str): 数据目录
            is_train (bool): 是否为训练集
            transform (callable, optional): 可选的数据转换
            blocks_count (int): 要拆分成的块数量
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        self.blocks_count = blocks_count
        
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
        
        # 获取数据形状信息
        if len(self.complete_files) > 0:
            sample_path = os.path.join(self.base_dir, self.complete_files[0])
            sample_data = np.load(sample_path, mmap_mode='r')
            self.data_shape = sample_data.shape
            print(f"原始数据形状: {self.data_shape}")
            
            # 计算每个块的切片数
            self.depth = self.data_shape[2]
            self.slices_per_block = self.depth // blocks_count
            if self.depth % blocks_count != 0:
                print(f"警告: 深度 {self.depth} 不能被块数 {blocks_count} 整除")
                self.slices_per_block = math.ceil(self.depth / blocks_count)
                
            print(f"将拆分为 {blocks_count} 个块，每个块包含 {self.slices_per_block} 个切片")
        
    def __len__(self):
        return len(self.complete_files)
    
    def __getitem__(self, idx):
        # 获取文件路径
        complete_path = os.path.join(self.base_dir, self.complete_files[idx])
        incomplete_path = os.path.join(self.base_dir, self.incomplete_files[idx])
        
        # 加载数据
        complete_data = np.load(complete_path)
        incomplete_data = np.load(incomplete_path)
        
        # 将数据拆分成块
        complete_blocks = []
        incomplete_blocks = []
        
        for i in range(self.blocks_count):
            start_slice = i * self.slices_per_block
            end_slice = min((i + 1) * self.slices_per_block, self.depth)
            
            # 提取当前块
            complete_block = complete_data[:, :, start_slice:end_slice]
            incomplete_block = incomplete_data[:, :, start_slice:end_slice]
            
            # 确保所有块大小一致 (填充最后一个块如果需要)
            if complete_block.shape[2] < self.slices_per_block:
                pad_width = self.slices_per_block - complete_block.shape[2]
                complete_block = np.pad(complete_block, ((0, 0), (0, 0), (0, pad_width)), 'constant')
                incomplete_block = np.pad(incomplete_block, ((0, 0), (0, 0), (0, pad_width)), 'constant')
            
            # 转换为张量
            complete_block = torch.from_numpy(complete_block).float()
            incomplete_block = torch.from_numpy(incomplete_block).float()
            
            # 应用变换（如果有）
            if self.transform:
                complete_block = self.transform(complete_block)
                incomplete_block = self.transform(incomplete_block)
            
            complete_blocks.append(complete_block)
            incomplete_blocks.append(incomplete_block)
        
        # 将块组合成批量
        complete_blocks = torch.stack(complete_blocks)  # [blocks_count, H, W, slices_per_block]
        incomplete_blocks = torch.stack(incomplete_blocks)  # [blocks_count, H, W, slices_per_block]
        
        return incomplete_blocks, complete_blocks

# ---------------------------
# 模型组件
# ---------------------------

class ConvBlock(nn.Module):
    """卷积块，包含两个卷积层和归一化"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetEncoder(nn.Module):
    """UNet编码器"""
    def __init__(self, in_channels, init_features=32):
        super(UNetEncoder, self).__init__()
        features = init_features
        
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = ConvBlock(features * 4, features * 8)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        return enc4, (enc1, enc2, enc3)

class UNetDecoder(nn.Module):
    """UNet解码器"""
    def __init__(self, out_channels, init_features=32):
        super(UNetDecoder, self).__init__()
        features = init_features
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features * 2, features)
        
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x, enc_features):
        enc1, enc2, enc3 = enc_features
        
        dec3 = self.upconv3(x)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

class SliceUNet(nn.Module):
    """处理单个切片的UNet模型"""
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(SliceUNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, init_features)
        self.decoder = UNetDecoder(out_channels, init_features)
    
    def forward(self, x):
        # 输入: [B, C, H, W]
        enc_out, enc_features = self.encoder(x)
        dec_out = self.decoder(enc_out, enc_features)
        return dec_out

class TransformerBlock(nn.Module):
    """标准Transformer Block"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [seq_len, batch, embed_dim]
        att_output, _ = self.att(x, x, x)
        out1 = self.norm1(x + self.dropout(att_output))
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_output)
        return out2

class SliceTransformer(nn.Module):
    """处理一个块中多个切片的Transformer模型"""
    def __init__(self, embed_dim, num_heads=4, ff_dim=512, num_layers=2, dropout=0.1):
        super(SliceTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # 输入: [slices, batch, features]
        for layer in self.layers:
            x = layer(x)
        return x

# ---------------------------
# 主模型架构
# ---------------------------

class BlockwiseSinogramTransformer(nn.Module):
    """
    块状正弦图变换器模型
    将大型正弦图拆分成块，分别处理，然后使用Transformer关联
    """
    def __init__(self, 
                 sinogram_shape=(182, 365, 1764), 
                 block_size=42,
                 init_features=32, 
                 embed_dim=256, 
                 num_heads=4, 
                 transformer_layers=2):
        super(BlockwiseSinogramTransformer, self).__init__()
        
        # 参数保存
        self.sinogram_shape = sinogram_shape
        self.height, self.width, self.depth = sinogram_shape
        self.block_size = block_size
        self.slices_per_block = self.depth // block_size
        
        # 确保形状合适
        if self.depth % block_size != 0:
            print(f"警告: 深度 {self.depth} 不能被块数 {block_size} 整除")
            self.slices_per_block = math.ceil(self.depth / block_size)
        
        # 每个切片的UNet（所有块和切片共享参数）
        self.slice_unet = SliceUNet(in_channels=1, out_channels=1, init_features=init_features)
        
        # Transformer特征提取所需的卷积层
        self.feature_extractor = nn.Conv2d(1, embed_dim, kernel_size=3, stride=2, padding=1)
        
        # 每个块内的切片Transformer
        self.slice_transformer = SliceTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim*2,
            num_layers=transformer_layers,
            dropout=0.1
        )
        
        # 最终输出投影
        self.output_projection = nn.Conv2d(embed_dim, 1, kernel_size=3, stride=1, padding=1)
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def _process_block(self, block):
        """处理单个块中的所有切片"""
        # block shape: [B, H, W, slices_per_block]
        batch_size, height, width, slices = block.shape
        
        # 将所有切片独立送入UNet
        slices_features = []
        
        for s in range(slices):
            # 提取单个切片
            slice_data = block[:, :, :, s].unsqueeze(1)  # [B, 1, H, W]
            
            # UNet处理切片
            unet_out = self.slice_unet(slice_data)  # [B, 1, H, W]
            
            # 特征提取
            features = self.feature_extractor(unet_out)  # [B, embed_dim, H/2, W/2]
            
            # 将特征平面变成向量
            h, w = features.shape[2], features.shape[3]
            features = features.view(batch_size, self.slice_transformer.embed_dim, -1)  # [B, embed_dim, H*W/4]
            features = features.permute(2, 0, 1)  # [H*W/4, B, embed_dim]
            
            slices_features.append(features)
        
        # 组合所有切片特征
        combined_features = torch.stack(slices_features, dim=0)  # [slices, H*W/4, B, embed_dim]
        seq_len, spatial_size, batch_size, embed_dim = combined_features.shape
        combined_features = combined_features.view(seq_len * spatial_size, batch_size, embed_dim)
        
        # 使用Transformer处理
        transformed = self.slice_transformer(combined_features)  # [slices*H*W/4, B, embed_dim]
        
        # 重新整形为每个切片的特征
        transformed = transformed.view(seq_len, spatial_size, batch_size, embed_dim)
        
        # 处理并返回输出
        output_slices = []
        
        for s in range(seq_len):
            # 获取当前切片的转换特征
            slice_features = transformed[s]  # [H*W/4, B, embed_dim]
            
            # 重新整形回空间形状
            slice_features = slice_features.permute(1, 2, 0)  # [B, embed_dim, H*W/4]
            slice_features = slice_features.view(batch_size, embed_dim, h, w)  # [B, embed_dim, H/2, W/2]
            
            # 投影回输出空间
            output = self.output_projection(slice_features)  # [B, 1, H/2, W/2]
            
            # 上采样到原始大小
            output = self.upsampler(output)  # [B, 1, H, W]
            
            output_slices.append(output.squeeze(1))  # [B, H, W]
        
        # 合并所有输出切片
        block_output = torch.stack(output_slices, dim=3)  # [B, H, W, slices]
        
        return block_output
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入正弦图块 [block_size, B, H, W, slices_per_block]
        
        Returns:
            output: 处理后的正弦图块 [block_size, B, H, W, slices_per_block]
        """
        block_size, batch_size, height, width, slices_per_block = x.shape
        output_blocks = []
        
        # 逐块处理
        for i in range(block_size):
            # 处理当前块
            block = x[i]  # [B, H, W, slices_per_block]
            output_block = self._process_block(block)  # [B, H, W, slices_per_block]
            output_blocks.append(output_block)
            
            # 显式清理，减轻内存压力
            torch.cuda.empty_cache()
        
        # 合并所有输出块
        output = torch.stack(output_blocks, dim=0)  # [block_size, B, H, W, slices_per_block]
        
        return output

# ---------------------------
# 训练函数
# ---------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, accumulation_steps=1, amp_enabled=True):
    """使用梯度累积和混合精度训练一个epoch"""
    from torch.cuda.amp import autocast, GradScaler
    
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    # 初始化梯度缩放器
    scaler = GradScaler(enabled=amp_enabled)
    
    # 清零梯度
    optimizer.zero_grad()
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 输入和目标形状：[B, block_size, H, W, slices_per_block]
            batch_size = inputs.size(0)
            block_size = inputs.size(1)
            
            # 重新排列为：[block_size, B, H, W, slices_per_block]
            inputs = inputs.permute(1, 0, 2, 3, 4)
            targets = targets.permute(1, 0, 2, 3, 4)
            
            # 将数据移至设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 使用混合精度进行前向传播和损失计算
            with autocast(enabled=amp_enabled):
                outputs = model(inputs)  # [block_size, B, H, W, slices_per_block]
                
                # 计算损失
                loss = criterion(outputs, targets) / accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 每 accumulation_steps 步更新一次
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 清理内存
                torch.cuda.empty_cache()
            
            # 更新统计信息
            losses.update(loss.item() * accumulation_steps, batch_size)
            
            # 在块级别计算指标
            with torch.no_grad():
                # 转换回 [B, block_size, H, W, slices_per_block] 形式
                outputs_cpu = outputs.detach().cpu().permute(1, 0, 2, 3, 4)
                targets_cpu = targets.cpu().permute(1, 0, 2, 3, 4)
                
                # 展平进行指标计算
                outputs_flat = outputs_cpu.reshape(-1, outputs_cpu.shape[-3], outputs_cpu.shape[-2], outputs_cpu.shape[-1])
                targets_flat = targets_cpu.reshape(-1, targets_cpu.shape[-3], targets_cpu.shape[-2], targets_cpu.shape[-1])
                
                # 计算指标
                from utils.metrics import calculate_metrics
                metrics = calculate_metrics(outputs_flat, targets_flat)
                psnr_meter.update(metrics['psnr'], batch_size * block_size)
                ssim_meter.update(metrics['ssim'], batch_size * block_size)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'psnr': f"{psnr_meter.avg:.2f}",
                'ssim': f"{ssim_meter.avg:.4f}",
                'gpu_mem': f"{torch.cuda.memory_reserved() / 1e9:.1f}GB"
            })
            
            # 定期显示内存状态
            if batch_idx % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"\nGPU内存: 分配={allocated:.2f}GB, 保留={reserved:.2f}GB")
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg

# ---------------------------
# 主函数
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BlockwiseSinogramTransformer model")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data/specific",
                        help="包含训练和测试数据的目录")
    
    # 模型参数
    parser.add_argument("--block_size", type=int, default=42,
                        help="将数据拆分成的块数量")
    parser.add_argument("--init_features", type=int, default=32,
                        help="UNet初始特征通道数")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Transformer嵌入维度")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Transformer头数")
    parser.add_argument("--transformer_layers", type=int, default=2,
                        help="Transformer层数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    
    # 优化参数
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader工作进程数")
    parser.add_argument("--pin_memory", action="store_true",
                        help="是否使用pin_memory")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./outputs/blockwise",
                        help="输出目录")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="保存模型的轮数间隔")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据集
    train_dataset = BlockwiseSinogramDataset(
        args.data_dir, 
        is_train=True, 
        blocks_count=args.block_size
    )
    
    # 分割训练集和验证集
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # 获取一个示例数据以确定形状
    sample_inputs, _ = next(iter(train_loader))
    _, _, height, width, slices_per_block = sample_inputs.shape
    
    # 实例化模型
    model = BlockwiseSinogramTransformer(
        sinogram_shape=(height, width, slices_per_block * args.block_size),
        block_size=args.block_size,
        init_features=args.init_features,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 训练循环
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        # 训练
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            accumulation_steps=args.accumulation_steps, amp_enabled=True
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
    
    print("Training completed!")

# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for inputs, targets in pbar:
                # 输入和目标形状：[B, block_size, H, W, slices_per_block]
                batch_size = inputs.size(0)
                block_size = inputs.size(1)
                
                # 重新排列为：[block_size, B, H, W, slices_per_block]
                inputs = inputs.permute(1, 0, 2, 3, 4)
                targets = targets.permute(1, 0, 2, 3, 4)
                
                # 将数据移至设备
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)  # [block_size, B, H, W, slices_per_block]
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 更新统计信息
                losses.update(loss.item(), batch_size)
                
                # 计算指标
                # 转换回 [B, block_size, H, W, slices_per_block] 形式
                outputs_cpu = outputs.cpu().permute(1, 0, 2, 3, 4)
                targets_cpu = targets.cpu().permute(1, 0, 2, 3, 4)
                
                # 展平进行指标计算
                outputs_flat = outputs_cpu.reshape(-1, outputs_cpu.shape[-3], outputs_cpu.shape[-2], outputs_cpu.shape[-1])
                targets_flat = targets_cpu.reshape(-1, targets_cpu.shape[-3], targets_cpu.shape[-2], targets_cpu.shape[-1])
                
                # 计算指标
                from utils.metrics import calculate_metrics
                metrics = calculate_metrics(outputs_flat, targets_flat)
                psnr_meter.update(metrics['psnr'], batch_size * block_size)
                ssim_meter.update(metrics['ssim'], batch_size * block_size)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{losses.avg:.4f}",
                    'psnr': f"{psnr_meter.avg:.2f}",
                    'ssim': f"{ssim_meter.avg:.4f}"
                })
                
                # 清理内存
                torch.cuda.empty_cache()
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg

# AverageMeter类用于统计平均值
class AverageMeter:
    """用于计算和存储平均值及当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    # 设置多进程启动方法
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    main()