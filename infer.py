# 推理文件：包含模型加载、特征提取和结果生成
import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from config import load_config, config as default_config

from model import EmbeddingModel

# 测试数据集类
class CowFaceTestDataset(torch.utils.data.Dataset):
    """
    牛脸测试数据集
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = [os.path.join(root_dir, fname)
                         for fname in sorted(os.listdir(root_dir))
                         if fname.endswith(('.jpg', '.jpeg', '.png'))]
        self.filenames = [os.path.basename(fname)
                         for fname in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        filename = self.filenames[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, filename

# 获取测试数据变换
def get_test_transform(image_size=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# 加载模型
def load_model(config, checkpoint_path="checkpoints/best_model115.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EmbeddingModel(
        backbone_name=config['model']['backbone'],
        embedding_size=config['model']['embedding_size']
    )
    model.to(device)
    
    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

# 提取特征
def extract_features(model, dataloader, device):
    """
    从数据加载器中提取所有图像的特征
    """
    all_embeddings = []
    all_filenames = []
    
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_filenames.extend(filenames)
    
    # 合并所有特征
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_filenames

# 生成提交文件
def generate_submission(embeddings, filenames, output_path="submission.csv", threshold=0.7):
    """
    生成提交文件，基于余弦相似度判断同一头牛
    - 第一列：测试图像对ID（从test.csv获取）
    - 第二列：0或1，表示是否是同一头牛
    - 第三列：相似度值
    """
    # 创建文件名到索引的映射，便于快速查找
    filename_to_idx = {os.path.splitext(fname)[0]: i for i, fname in enumerate(filenames)}
    
    # 读取test.csv文件获取图像对ID
    try:
        test_pairs_df = pd.read_csv('test-1118.csv')
        print(f"Loaded {len(test_pairs_df)} test image pairs")
    except Exception as e:
        print(f"Error reading test.csv: {e}")
        raise
    
    results = []
    
    # 计算所有图像对之间的余弦相似度矩阵
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # 处理每一个测试图像对
    for idx, row in test_pairs_df.iterrows():
        pair_id = row['ID_ID']  # 获取图像对ID
        img1_name, img2_name = pair_id.split('_')  # 分割两个图像名称
        
        # 查找两个图像在特征矩阵中的索引
        if img1_name in filename_to_idx and img2_name in filename_to_idx:
            img1_idx = filename_to_idx[img1_name]
            img2_idx = filename_to_idx[img2_name]
            
            # 获取两个图像之间的相似度
            similarity = similarity_matrix[img1_idx, img2_idx]
            
            # 根据阈值判断是否是同一头牛（1表示同一头牛，0表示不同）
            is_same = 1 if similarity > threshold else 0
            
            # 添加到结果列表
            results.append({
                'ID_ID': pair_id,        # 第一列：图像对ID
                'output': is_same,        # 第二列：是否同一头牛 (0/1)
                'similarity': similarity # 第三列：相似度值
            })
        else:
            # 如果找不到某个图像，记录为-1（表示未知）
            results.append({
                'ID_ID': pair_id,
                'output': -1,  # 未知
                'similarity': 0.0
            })
    
    # 创建DataFrame并保存为CSV
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
    print(f"Total pairs processed: {len(results)}")
    print(f"Pairs identified as same cow: {sum(1 for r in results if r['output'] == 1)}")
    print(f"Pairs identified as different cows: {sum(1 for r in results if r['output'] == 0)}")
    print(f"Pairs with unknown images: {sum(1 for r in results if r['output'] == -1)}")
    
    return submission_df


def main():
    parser = argparse.ArgumentParser(description='Cow Face Recognition Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/test-new',
                      help='Path to test data directory')
    parser.add_argument('--output', type=str, default='submission.csv',
                      help='Output submission file path')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Similarity threshold for cow identification')
    args = parser.parse_args()
    
    # 加载配置（不再需要传入配置文件路径）
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(config, args.checkpoint)
    
    # 准备测试数据
    transform = get_test_transform(config['data']['image_size'])
    test_dataset = CowFaceTestDataset(
        root_dir=args.test_dir,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 提取特征
    embeddings, filenames = extract_features(model, test_loader, device)
    
    # 生成提交文件
    generate_submission(embeddings, filenames, args.output, args.threshold)

if __name__ == '__main__':
    main()