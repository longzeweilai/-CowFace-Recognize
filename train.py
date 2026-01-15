# 训练文件：包含训练逻辑和难例挖掘
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import load_config, config as default_config
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib字体，支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from data_processing import TripletDataset, get_train_transform, get_all_images
from model import EmbeddingModel, get_model

# 难例挖掘函数（优化版）
def hard_triplet_mining(embeddings, labels, margin=0.3, device='cuda', config=None, max_samples=1000):
    """
    执行难例挖掘，返回难三元组的索引
    难三元组：a-p距离较大，a-n距离较小
    
    优化版本：
    1. 进一步限制参与难例挖掘的最大样本数
    2. 使用更高效的掩码计算方法
    3. 减少内存占用
    """
    n = embeddings.size(0)
    
    # 优化：进一步减小最大样本数，避免计算过大的距离矩阵
    max_samples = min(max_samples, 1000)  # 严格限制最大样本数为1000
    if n > max_samples:
        # 随机采样部分样本参与难例挖掘
        indices = torch.randperm(n)[:max_samples]
        embeddings = embeddings[indices]
        labels = labels[indices]
        n = embeddings.size(0)
    
    print(f"Processing {n} samples for hard triplet mining")
    
    # 计算所有embedding对之间的距离
    distances = torch.cdist(embeddings, embeddings, p=2).to(device)  # (n, n)
    
    # 创建相同标签的掩码
    mask_positive = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    mask_positive.diagonal().fill_(0)  # 排除自己与自己的匹配
    
    # 为每个锚点找到距离最大的正样本（难正样本）
    pos_distances = distances * mask_positive
    # 每个锚点保留较少的难正样本，减少内存占用
    num_hard_samples = min(2, pos_distances.size(1))  # 减少到最多2个难正样本
    hard_positive_distances, hard_positive_indices = pos_distances.topk(num_hard_samples, dim=1)
    
    # 为每个锚点找到距离最小的负样本（难负样本）
    mask_negative = 1 - mask_positive
    neg_distances = distances * mask_negative
    neg_distances[neg_distances == 0] = float('inf')
    # 每个锚点保留较少的难负样本
    hard_negative_distances, hard_negative_indices = neg_distances.topk(num_hard_samples, dim=1, largest=False)
    
    # 构建所有可能的三元组组合
    # 优化：使用列表预分配和限制三元组数量，避免内存溢出
    max_triplets = 1000  # 限制最大三元组数量
    anchor_indices = []
    positive_indices = []
    negative_indices = []
    
    # 逐批处理锚点，减少内存占用
    batch_size = 200  # 每批处理的锚点数
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        
        for i in range(batch_start, batch_end):
            # 只有当有正样本和负样本时才处理
            if mask_positive[i].sum() > 0 and mask_negative[i].sum() > 0:
                # 检查所有可能的正负样本组合
                for j in range(hard_positive_indices.size(1)):
                    for l in range(hard_negative_indices.size(1)):
                        # 难例条件：d(a,p) > d(a,n) - margin 或等价于 d(a,p) + margin > d(a,n)
                        if hard_positive_distances[i, j] + margin > hard_negative_distances[i, l]:
                            anchor_indices.append(i)
                            positive_indices.append(hard_positive_indices[i, j].item())
                            negative_indices.append(hard_negative_indices[i, l].item())
            
            # 优化：如果已经收集了足够的三元组，提前退出
            if len(anchor_indices) >= max_triplets:
                break
        
        if len(anchor_indices) >= max_triplets:
            break
    
    # 清理不再需要的大张量
    del distances, mask_positive, mask_negative, pos_distances, neg_distances
    torch.cuda.empty_cache()
    
    # 如果找到的有效三元组不足，返回随机三元组
    if len(anchor_indices) == 0:
        # 如果没有有效的难三元组，返回随机三元组
        min_hard_triplets = config['train']['min_hard_triplets'] if config and 'train' in config and 'min_hard_triplets' in config['train'] else 32
        return torch.arange(0, min(n, min_hard_triplets)), \
               torch.randint(0, n, (min(n, min_hard_triplets),)), \
               torch.randint(0, n, (min(n, min_hard_triplets),))
    
    # 限制返回的三元组数量
    if len(anchor_indices) > max_triplets:
        # 随机选择部分有效三元组
        indices = torch.randperm(len(anchor_indices))[:max_triplets]
        anchor_indices = [anchor_indices[i] for i in indices]
        positive_indices = [positive_indices[i] for i in indices]
        negative_indices = [negative_indices[i] for i in indices]
    
    print(f"Generated {len(anchor_indices)} hard triplets")
    
    # 返回有效的难三元组索引
    return torch.tensor(anchor_indices), torch.tensor(positive_indices), torch.tensor(negative_indices)

# 获取数据加载器
def get_dataloader(config):
    transform = get_train_transform(config['data']['image_size'])
    dataset = TripletDataset(root=config['data']['train_dir'], transform=transform)
    
    # 优化点11：自适应调整num_workers，避免CPU过载
    # 根据系统CPU核心数量动态调整工作线程数
    import multiprocessing
    available_cpus = multiprocessing.cpu_count()
    optimal_workers = min(config['data']['num_workers'], available_cpus // 2)  # 最多使用一半的CPU核心
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=optimal_workers,  # 使用优化后的工作线程数
        pin_memory=True,
        drop_last=True,
        # 优化点12：添加预取缓冲区，提高数据加载效率
        prefetch_factor=2  # 每个worker预先加载2个batch
    )
    return dataloader, dataset

# 获取模型
def get_model(config):
    model = EmbeddingModel(
        backbone_name=config['model']['backbone'],
        embedding_size=config['model']['embedding_size']
    )
    return model.cuda() if torch.cuda.is_available() else model

def main():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 优化点13：设置CUDA基准测试，对固定大小输入加速
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # 对于固定大小的输入，启用benchmark可以提高速度

    # 数据加载
    train_loader, train_dataset = get_dataloader(config)
    
    # 为难例挖掘准备数据（可选：可以在每个epoch后更新）
    use_hard_mining = config['train'].get('hard_mining', True)
    print(f"Hard triplet mining: {use_hard_mining}")

    # 模型
    model = get_model(config)
    model.to(device)
    
    # 优化点14：梯度累积参数设置
    accumulation_steps = 2  # 累积2个batch的梯度再更新参数
    
    # 优化器（分层学习率）
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['train']['lr'] * 0.1},
        {'params': model.neck.parameters(), 'lr': config['train']['lr']}
    ], weight_decay=config['train']['weight_decay'])

    # 学习率调度
    # 优化点15：使用预热+余弦退火的学习率调度
    from torch.optim.lr_scheduler import OneCycleLR
    max_lr = config['train']['lr']
    total_steps = config['train']['epochs'] * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        total_steps=total_steps, 
        pct_start=0.1,  # 预热占总步数的10%
        div_factor=10,  # 初始学习率为max_lr的1/10
        final_div_factor=1000,  # 最终学习率为max_lr的1/1000
        anneal_strategy='cos'
    )

    # 损失函数
    margin = config['train'].get('margin', 0.3)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    # TensorBoard
    writer = SummaryWriter(log_dir='logs')

    # 开始训练
    epochs = config['train']['epochs']
    best_loss = float('inf')
    
    # 存储loss和学习率历史，用于绘图
    loss_history = []
    lr_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        # 每隔几个epoch进行一次难例挖掘
        if use_hard_mining and (epoch % 5 == 0 or epoch == 0):  # 优化点7：降低难例挖掘频率，每5个epoch一次
            print("Performing hard triplet mining...")
            # 优化：直接通过get_all_images的max_samples参数限制样本数量，避免一次性加载全部图像
            max_mining_samples = 3000  # 减小样本数量，降低内存占用
            all_images, all_labels, _ = get_all_images(train_dataset, max_samples=max_mining_samples)
            print(f"Using {len(all_images)} samples for hard mining")
            
            # 分批将图像转移到GPU并计算embeddings，避免一次性占用过多GPU内存
            all_images = all_images.to(device)
            all_labels = all_labels.to(device)
            
            # 计算所有图像的embedding
            model.eval()
            with torch.no_grad():
                all_embeddings = []
                # 分批处理以避免内存溢出，从配置中获取batch_size
                # 优化点9：增加难例挖掘的批次大小，提高处理效率
                hard_embedding_batch_size = min(config['train']['hard_embedding_batch_size'] * 2, 64)  # 增大批次但限制最大值
                for i in range(0, len(all_images), hard_embedding_batch_size):
                    batch_embeddings = model(all_images[i:i+hard_embedding_batch_size])
                    all_embeddings.append(batch_embeddings)
                all_embeddings = torch.cat(all_embeddings, dim=0)
            model.train()
            
            # 挖掘难三元组，传入max_samples参数控制计算规模
            anchor_indices, pos_indices, neg_indices = hard_triplet_mining(
                all_embeddings, all_labels, margin=margin, device=device, config=config, max_samples=2000
            )
                
            print(f"Found {len(anchor_indices)} hard triplets")
                
            # 使用难三元组进行额外的训练迭代
            hard_batch_size = config['train']['hard_train_batch_size']
            # 优化点10：限制难例训练的最大迭代次数，避免训练过久
            num_hard_iterations = min(50, len(anchor_indices) // hard_batch_size)  # 从100降低到50次迭代
                
            # 难例训练也使用梯度累积
            hard_accumulation_steps = min(accumulation_steps, 4)  # 难例训练可以使用更大的累积步数
                
            for i in range(num_hard_iterations):
                start_idx = i * hard_batch_size
                end_idx = min(start_idx + hard_batch_size, len(anchor_indices))
                
                a_idx = anchor_indices[start_idx:end_idx]
                p_idx = pos_indices[start_idx:end_idx]
                n_idx = neg_indices[start_idx:end_idx]
                
                anchor = all_images[a_idx]
                pos = all_images[p_idx]
                neg = all_images[n_idx]
                
                # 使用梯度累积进行难例训练
                if i % hard_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                a_emb = model(anchor)
                p_emb = model(pos)
                n_emb = model(neg)
                
                hard_loss = criterion(a_emb, p_emb, n_emb)
                hard_loss = hard_loss / hard_accumulation_steps  # 缩放损失
                hard_loss.backward()
                
                # 只有在累积结束或最后一次迭代时更新参数
                if (i + 1) % hard_accumulation_steps == 0 or i == num_hard_iterations - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # 更新学习率
                    for _ in range(hard_accumulation_steps):
                        scheduler.step()
                
                running_loss += hard_loss.item() * hard_accumulation_steps  # 还原真实损失
        
        # 正常批量训练
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for i, (anchor, pos, neg) in enumerate(progress_bar):
            anchor, pos, neg = anchor.to(device, non_blocking=True), pos.to(device, non_blocking=True), neg.to(device, non_blocking=True)
            
            # 优化点16：梯度累积
            # 只在累积开始时清零梯度
            if i % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True节省内存
            
            # 优化点17：前向传播和损失计算
            a_emb = model(anchor)
            p_emb = model(pos)
            n_emb = model(neg)
            
            loss = criterion(a_emb, p_emb, n_emb)
            loss = loss / accumulation_steps  # 梯度累积时需要缩放损失
            
            # 反向传播
            loss.backward()
            
            # 优化点18：梯度裁剪
            if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 优化点19：更新学习率
                scheduler.step()
            
            running_loss += loss.item() * accumulation_steps  # 乘以累积步数还原真实损失
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

        epoch_loss = running_loss / (len(train_loader) + (num_hard_iterations if use_hard_mining and (epoch % 5 == 0 or epoch == 0) else 0))
        epoch_time = time.time() - start_time

        # 记录日志
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存loss和学习率历史
        loss_history.append(epoch_loss)
        lr_history.append(optimizer.param_groups[0]['lr'])

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.0f}s")

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # 确保保存目录存在
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"Best model saved with loss: {best_loss:.4f}")

        # 移除重复的学习率更新，因为已经在训练循环中更新过了

    # 绘制并保存loss和学习率变化曲线
    os.makedirs('logs/plots', exist_ok=True)
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-', color='b')
    plt.title('训练损失变化曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('logs/plots/loss_curve.png', dpi=300, bbox_inches='tight')
    
    # 绘制学习率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), lr_history, marker='s', linestyle='-', color='r')
    plt.title('学习率变化曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('logs/plots/lr_curve.png', dpi=300, bbox_inches='tight')
    
    # 绘制双Y轴图，同时显示loss和学习率
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.subplot(111)
    ax1.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-', color='b')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs+1), lr_history, marker='s', linestyle='-', color='r')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')  # 学习率使用对数刻度
    
    plt.title('损失和学习率变化曲线')
    plt.tight_layout()
    plt.savefig('logs/plots/loss_lr_curve.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print("Loss和学习率曲线已保存到logs/plots目录")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    print("Training finished.")
    print(f"Best Loss: {best_loss:.4f}")
    writer.close()

if __name__ == '__main__':
    main()