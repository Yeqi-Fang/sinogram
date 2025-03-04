"""
Implementation of SinogramTransformer model for incomplete ring PET reconstruction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinogramBlockEncoder(nn.Module):
    """处理单个环差块的编码器，利用adaptive pooling降低token尺寸"""
    def __init__(self, in_channels, out_channels=128, pool_size=(4, 4)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pool_size = pool_size
        
    def forward(self, x):
        # x: (B, H, W, C) -> convert to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)  # [B, out_channels, H//4, W//4]
        # Instead of flattening the entire spatial map, adaptively pool to a fixed (small) size.
        x = F.adaptive_avg_pool2d(x, self.pool_size)  # [B, out_channels, pool_size[0], pool_size[1]]
        return x.flatten(2)  # [B, out_channels, pool_size[0]*pool_size[1]]

class SinogramBlockDecoder(nn.Module):
    """处理单个环差块的解码器，知道输入来自固定池化尺寸"""
    def __init__(self, in_channels, out_channels, pool_size=(4, 4)):
        super().__init__()
        self.pool_size = pool_size  # (pH, pW) used in encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
    def forward(self, x, original_shape):
        # x: [B, in_channels, pool_size[0]*pool_size[1]]
        # Reshape x back to a spatial map with the reduced dimensions.
        x = x.view(x.shape[0], -1, self.pool_size[0], self.pool_size[1])
        x = self.decoder(x)
        # Finally, interpolate to match the original spatial dimensions.
        x = F.interpolate(x, size=original_shape, mode='bilinear', align_corners=False)
        # Return to shape [B, H, W, C]
        return x.permute(0, 2, 3, 1)

class SinogramTransformer(nn.Module):
    def __init__(self, 
                 sinogram_shape,    # (H, W, total_channels)
                 block_channels,    # 每个环差块的通道数
                 embed_dim=128, 
                 num_heads=8,
                 pool_size=(4, 4)):
        super().__init__()
        self.height, self.width, self.total_channels = sinogram_shape
        self.block_channels = block_channels
        self.num_blocks = self.total_channels // self.block_channels
        self.pool_size = pool_size
        
        # 编码器和解码器（都使用自定义的pool_size）
        self.block_encoder = SinogramBlockEncoder(
            in_channels=self.block_channels,
            out_channels=embed_dim,
            pool_size=pool_size
        )
        self.block_decoder = SinogramBlockDecoder(
            in_channels=embed_dim,
            out_channels=self.block_channels,
            pool_size=pool_size
        )
        
        # Now, the encoder outputs tokens of shape:
        # [B, embed_dim, pool_size[0]*pool_size[1]]
        # So we define d_model as:
        d_model = embed_dim * (pool_size[0] * pool_size[1])
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=4
        )
        
        # 位置编码：对每个块生成一个learnable embedding
        self.position_embed = nn.Embedding(
            self.num_blocks, 
            d_model
        )
        
    def forward(self, x):
        # x: [B, H, W, Total_Channels]
        B = x.shape[0]
        original_shape = (self.height, self.width)
        
        # 分割成块: [B, num_blocks, H, W, block_channels]
        blocks = x.view(B, self.height, self.width, self.num_blocks, self.block_channels)
        blocks = blocks.permute(0, 3, 1, 2, 4)  # [B, num_blocks, H, W, block_channels]
        blocks = blocks.reshape(B * self.num_blocks, self.height, self.width, self.block_channels)
        
        # 编码每个块，得到大小为 [B*num_blocks, embed_dim, pool_size[0]*pool_size[1]]
        encoded = self.block_encoder(blocks)
        # 变回 (B, num_blocks, d_model)
        encoded = encoded.view(B, self.num_blocks, -1)
        
        # 添加位置编码（广播到每个batch）
        positions = torch.arange(self.num_blocks, device=x.device).expand(B, self.num_blocks)
        encoded = encoded + self.position_embed(positions)
        
        # Transformer处理: 输出仍为 [B, num_blocks, d_model]
        transformed = self.transformer(encoded)
        
        # 解码每个块
        decoded = []
        for i in range(self.num_blocks):
            # 提取第i个块的token并恢复形状 [B, d_model] -> [B, embed_dim, pool_size[0]*pool_size[1]]
            block_token = transformed[:, i].view(B, -1, self.pool_size[0] * self.pool_size[1])
            block_decoded = self.block_decoder(block_token, original_shape)  # [B, H, W, block_channels]
            decoded.append(block_decoded)
            
        # 合并所有块：最终输出 shape [B, H, W, Total_Channels]
        output = torch.cat(decoded, dim=-1)
        return output

# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义sinogram的维度
    height = 112
    width = 225
    total_channels = 1024
    block_channels = 32
    
    # 模拟输入数据 [B, H, W, total_channels]
    dummy_input = torch.randn(1, height, width, total_channels).to(device)
    
    # 初始化模型。注意，pool_size越小越省内存，但可能会损失部分空间细节
    model = SinogramTransformer(
        sinogram_shape=(height, width, total_channels),
        block_channels=block_channels,
        embed_dim=128,
        num_heads=8,
        pool_size=(4, 4)  # 可以尝试 (2,2), (4,4)等不同尺寸
    ).to(device)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")