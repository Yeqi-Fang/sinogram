"""
生成特定尺寸的合成正弦图数据，专门用于PET重建模型测试。
固定高度为112，宽度为225，通道数为1024，块大小为32。
生成100张训练图像和10张测试图像，每张图像都有一个角度被掩码处理。
使用多进程加速数据生成，避免GPU加速。
"""
import os
import numpy as np
import math
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
# 设置全局参数
HEIGHT = 112
WIDTH = 225
DEPTH = 1024
BLOCK_SIZE = 32

def create_angle_mask(sinogram_shape, start_angle, end_angle):
    """
    创建角度掩码
    """
    total_angles = sinogram_shape[0]
    mask = np.ones(sinogram_shape)
    
    # 将角度转换为索引
    start_idx = int(start_angle / 180 * total_angles)
    end_idx = int(end_angle / 180 * total_angles)
    
    # 将掩码区域设置为0
    mask[start_idx:end_idx, :, :] = 0
    
    return mask

def apply_mask(sinogram, mask):
    """
    应用掩码
    """
    return sinogram * mask

def normalize_sinogram(sinogram):
    """
    归一化正弦图数据到[0, 1]范围
    """
    min_val = np.min(sinogram)
    max_val = np.max(sinogram)
    return (sinogram - min_val) / (max_val - min_val + 1e-8)

def generate_phantom_sinogram(height, width, depth, seed=None, noise_level=0.02):
    """
    生成带有点源的合成正弦图，使用固定种子以便重现
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 创建空的正弦图
    sinogram = np.zeros((height, width, depth), dtype=np.float32)
    
    # 生成随机点源
    num_sources = 5
    sources = []
    for _ in range(num_sources):
        # 图像空间中的随机位置
        x = np.random.uniform(-0.75, 0.75)
        y = np.random.uniform(-0.75, 0.75)
        # 随机强度（中心位置更高）
        intensity = np.random.uniform(0.5, 1.0) * (1.0 - 0.5 * (x**2 + y**2))
        # 随机大小
        size = np.random.uniform(0.05, 0.2)
        sources.append((x, y, intensity, size))
    
    # 创建角度和径向位置数组
    angles = np.linspace(0, math.pi, height)
    r_values = np.linspace(-1, 1, width)
    
    # 创建深度缩放数组
    depth_scale = np.linspace(1.0, 0.4, depth)
    depth_indices = np.arange(depth)
    ring_diff = np.abs(depth_indices - depth // 2) / (depth // 2)
    depth_falloff = np.exp(-ring_diff * 2)
    
    # 对每个角度
    for angle_idx, angle in enumerate(angles):
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        
        # 对每个径向位置
        for r_idx, r in enumerate(r_values):
            
            # 对每个源，计算贡献
            for x, y, intensity, size in sources:
                # 投影点到线上
                proj_dist = abs(x * cos_theta + y * sin_theta - r)
                
                # 计算贡献（高斯曲线）
                contrib = intensity * np.exp(-(proj_dist**2) / (2 * size**2))
                
                # 添加到正弦图（使用向量化操作应用到所有深度）
                sinogram[angle_idx, r_idx, :] += contrib * depth_scale * depth_falloff
    
    # 添加背景活动
    background = np.ones_like(sinogram) * 0.05
    sinogram = sinogram + background
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, sinogram.shape)
    sinogram = sinogram + noise
    
    # 确保非负值
    sinogram = np.clip(sinogram, 0.0, None)
    
    # 归一化
    sinogram = normalize_sinogram(sinogram)
    
    # 确保float32类型
    return sinogram.astype(np.float32)

def generate_brain_like_sinogram(height, width, depth, seed=None, noise_level=0.02):
    """
    生成类似大脑活动的合成正弦图，使用固定种子以便重现
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 创建空的正弦图
    sinogram = np.zeros((height, width, depth), dtype=np.float32)
    
    # 定义类脑结构
    structures = []
    
    # 外部颅骨（环）
    structures.append(('ring', 0, 0, 0.8, 0.15, 0.4))  # 类型，cx，cy，半径，厚度，强度
    
    # 脑组织（填充圆）
    structures.append(('disk', 0, 0, 0.65, 0, 0.6))  # 类型，cx，cy，半径，_，强度
    
    # 脑室（较暗区域）
    structures.append(('disk', 0.1, 0, 0.15, 0, -0.3))  # 负强度用于减法
    structures.append(('disk', -0.1, 0, 0.15, 0, -0.3))
    
    # 热点（小亮区）
    for _ in range(10):
        # 大脑内的随机位置
        angle = np.random.uniform(0, 2 * math.pi)
        dist = np.random.uniform(0, 0.5)
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        # 随机大小和强度
        size = np.random.uniform(0.05, 0.12)
        intensity = np.random.uniform(0.3, 0.7)
        structures.append(('disk', x, y, size, 0, intensity))
    
    # 创建深度缩放数组
    depth_indices = np.arange(depth)
    ring_diff = np.abs(depth_indices - depth // 2) / (depth // 2)
    depth_falloff = np.exp(-ring_diff * 2)
    
    # 创建角度和径向位置数组
    angles = np.linspace(0, math.pi, height)
    r_values = np.linspace(-1, 1, width)
    
    # 对每个角度
    for angle_idx, angle in tqdm(enumerate(angles), desc="生成类脑正弦图", total=len(angles)):
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        
        # 对每个径向位置
        for r_idx, r in enumerate(r_values):
            
            # 处理每个结构
            for structure in structures:
                struct_type, cx, cy, radius, thickness, intensity = structure
                
                # 将中心投影到线上
                center_proj = cx * cos_theta + cy * sin_theta
                
                if struct_type == 'ring':
                    # 计算环投影的起点和终点
                    inner_radius = radius - thickness/2
                    outer_radius = radius + thickness/2
                    
                    # 内部和外部边缘投影
                    inner_start = center_proj - inner_radius
                    inner_end = center_proj + inner_radius
                    outer_start = center_proj - outer_radius
                    outer_end = center_proj + outer_radius
                    
                    # 检查当前r是否在投影环内
                    if (outer_start <= r <= inner_start) or (inner_end <= r <= outer_end):
                        # r到最近内边缘的距离
                        if r <= center_proj:
                            edge_dist = min(abs(r - outer_start), abs(r - inner_start))
                        else:
                            edge_dist = min(abs(r - inner_end), abs(r - outer_end))
                        
                        # 强度曲线
                        profile = intensity * (1 - edge_dist / thickness)
                        
                        # 一次添加到所有深度
                        if profile > 0:
                            sinogram[angle_idx, r_idx, :] += profile * depth_falloff
                
                elif struct_type == 'disk':
                    # 从投影线到中心的距离
                    dist_to_center = abs(center_proj - r)
                    
                    # 检查当前r是否在投影盘内
                    if dist_to_center <= radius:
                        # 根据到中心的距离计算曲线
                        edge_factor = 1 - (dist_to_center / radius)**2
                        profile = intensity * edge_factor
                        
                        # 一次添加到所有深度
                        sinogram[angle_idx, r_idx, :] += profile * depth_falloff
    
    # 添加背景活动
    background = np.ones_like(sinogram) * 0.1
    sinogram = sinogram + background
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, sinogram.shape)
    sinogram = sinogram + noise
    
    # 确保非负值
    sinogram = np.clip(sinogram, 0.0, None)
    
    # 归一化
    sinogram = normalize_sinogram(sinogram)
    
    # 确保float32类型
    return sinogram.astype(np.float32)

def visualize_slice(args):
    """
    可视化正弦图的单个切片并保存到文件
    
    Args:
        args: 元组包含 (sinogram, slice_idx, title, output_path, cmap)
    """
    sinogram, slice_idx, title, output_path, cmap = args
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sinogram[:, :, slice_idx], cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_parallel(sinogram, slice_indices, output_dir, prefix=""):
    """
    使用多进程并行可视化正弦图切片
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数
    viz_args = []
    for slice_idx in slice_indices:
        output_path = os.path.join(output_dir, f"{prefix}sinogram_slice_{slice_idx}.png")
        title = f"{prefix.capitalize()}Sinogram (Slice {slice_idx})"
        viz_args.append((sinogram, slice_idx, title, output_path, 'hot'))
    
    # 使用进程池并行处理可视化
    with ProcessPoolExecutor(max_workers=min(len(slice_indices), os.cpu_count())) as executor:
        futures = [executor.submit(visualize_slice, arg) for arg in viz_args]
        
        # 等待所有可视化完成
        for future in as_completed(futures):
            # 这将引发执行期间发生的任何异常
            try:
                path = future.result()
                print(f"已保存: {path}")
            except Exception as e:
                print(f"可视化出错: {e}")

def generate_sample(args):
    """
    生成单个样本，用于并行处理
    
    Args:
        args: 元组包含 (idx, is_train, pattern, noise_level, mask, output_dir, visualize)
    
    Returns:
        元组 (idx, complete_path, incomplete_path)
    """
    idx, is_train, pattern, noise_level, mask, output_dir, visualize = args
    
    # 确定输出路径
    target_dir = os.path.join(output_dir, "train" if is_train else "test")
    
    # 为每个样本使用不同种子
    seed = idx * 100 if is_train else (idx + 1) * 1000 + 100
    
    # 选择生成器函数
    generator_func = generate_brain_like_sinogram if pattern == "brain" else generate_phantom_sinogram
    
    try:
        # 生成完整正弦图
        complete_sinogram = generator_func(
            HEIGHT, WIDTH, DEPTH, 
            seed=seed,
            noise_level=noise_level
        )
        
        # 生成不完整正弦图
        incomplete_sinogram = apply_mask(complete_sinogram, mask)
        
        # 保存为.npy格式
        complete_path = os.path.join(target_dir, f"complete_{idx+1}.npy")
        incomplete_path = os.path.join(target_dir, f"incomplete_{idx+1}.npy")
        
        np.save(complete_path, complete_sinogram)
        np.save(incomplete_path, incomplete_sinogram)
        
        # 可视化特定样本
        if visualize and ((is_train and (idx == 0 or idx == 49 or idx == 99)) or 
                          (not is_train and idx == 0)):
            vis_dir = os.path.join(output_dir, "visualizations", 
                                   f"{'train' if is_train else 'test'}_sample_{idx+1}")
            
            # 选择可视化切片
            slice_indices = [0, DEPTH//8, DEPTH//4, DEPTH//2, 3*DEPTH//4, DEPTH-1]
            slice_indices = [min(j, DEPTH-1) for j in slice_indices]
            
            # 并行生成可视化
            visualize_parallel(complete_sinogram, slice_indices, vis_dir, prefix="complete_")
            visualize_parallel(incomplete_sinogram, slice_indices, vis_dir, prefix="incomplete_")
        
        return idx, complete_path, incomplete_path
    
    except Exception as e:
        print(f"生成样本 {idx+1} 出错: {e}")
        return idx, None, None

def main():
    parser = argparse.ArgumentParser(description="生成特定尺寸的合成正弦图数据集")
    parser.add_argument("--output_dir", type=str, default="./data/specific",
                       help="输出目录")
    parser.add_argument("--pattern", type=str, default="brain",
                       choices=["phantom", "brain"],
                       help="生成模式")
    parser.add_argument("--noise_level", type=float, default=0.02,
                       help="噪声级别")
    parser.add_argument("--mask_angle_start", type=float, default=30,
                       help="掩码起始角度（度）")
    parser.add_argument("--mask_angle_end", type=float, default=60,
                       help="掩码结束角度（度）")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="生成可视化")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="用于数据生成的进程数，默认为CPU核心数")
    
    args = parser.parse_args()
    
    # 固定参数
    num_train = 100
    num_test = 10
    
    # 设置进程数
    num_processes = args.num_processes if args.num_processes else multiprocessing.cpu_count()
    print(f"使用 {num_processes} 个进程进行数据生成")
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建掩码（所有图像使用相同的掩码）
    mask = create_angle_mask(
        (HEIGHT, WIDTH, DEPTH), 
        args.mask_angle_start, 
        args.mask_angle_end
    )
    
    start_time = time.time()
    print(f"开始生成数据集，训练: {num_train}个样本，测试: {num_test}个样本")
    print(f"图像尺寸: ({HEIGHT}, {WIDTH}, {DEPTH})")
    
    # 准备训练样本参数
    train_args = [(i, True, args.pattern, args.noise_level, mask, output_dir, args.visualize) 
                 for i in range(num_train)]
    
    # 准备测试样本参数
    test_args = [(i, False, args.pattern, args.noise_level, mask, output_dir, args.visualize) 
                for i in range(num_test)]
    
    # 合并所有参数
    all_args = train_args + test_args
    
    # 使用进程池并行生成所有样本
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 提交所有任务
        futures = [executor.submit(generate_sample, arg) for arg in all_args]
        
        # 显示进度条
        with tqdm(total=len(all_args), desc="生成数据") as pbar:
            for future in as_completed(futures):
                idx, complete_path, incomplete_path = future.result()
                pbar.update(1)
    
    # 记录时间和结果
    total_time = time.time() - start_time
    print(f"\n完成！总处理时间：{total_time:.2f}秒")
    print(f"数据保存到：{output_dir}")
    print(f"- 训练样本：{num_train}个（在{train_dir}）")
    print(f"- 测试样本：{num_test}个（在{test_dir}）")
    print("所有文件使用.npy格式保存，数据类型为float32")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows下多进程支持
    main()