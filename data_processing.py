# 数据处理文件：包含数据集定义和数据变换
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from collections import defaultdict
import os
import random

# 训练在线数据增强效果比离线好些
def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),  # 先放大再裁剪
        transforms.RandomCrop(image_size),  # 随机裁剪到目标尺寸
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # 小角度旋转
        transforms.RandomAffine(
            degrees=0,  # 不额外旋转
            translate=(0.05, 0.05),  # 轻微平移
            scale=(0.95, 1.05)  # 轻微缩放
        ),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 测试数据标准化和训练时一样但不增强
def get_test_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 训练数据集：带难例挖掘的三元组数据集
class TripletDataset(Dataset):
    def __init__(self, root, transform=None, hard_mining=False):
        self.root = root
        self.transform = transform
        self.hard_mining = hard_mining
        
        # 使用 ImageFolder 加载数据
        dataset = datasets.ImageFolder(root=root, transform=None)
        self.samples = dataset.samples  # [(path, class_idx), ...]
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        
        # 按类别分组
        self.class_to_images = defaultdict(list)
        for path, class_idx in self.samples:
            self.class_to_images[class_idx].append(path)
        
        # 只保留有 ≥2 图片的类别用于构造正样本对
        self.valid_classes = [c for c, imgs in self.class_to_images.items() if len(imgs) >= 2]
        self.all_classes = list(self.class_to_images.keys())
        
        # PIL transform（在 __getitem__ 中应用）
        self.loader = datasets.folder.default_loader
        if transform is None:
            self.transform = get_train_transform()

        print(f"Found {len(self.all_classes)} cow identities.")
        print(f"{len(self.valid_classes)} cows have ≥2 images (can form positive pairs).")
        print(f"Total images: {len(self.samples)}")

    def __len__(self):
        # 设定一个合理的 epoch 大小
        return 10000  # 可调

    def __getitem__(self, _):
        # 随机选一个有多个图像的类作为 anchor/positive 来源
        anchor_class = random.choice(self.valid_classes)
        anchor_path = random.choice(self.class_to_images[anchor_class])
        pos_path = random.choice([p for p in self.class_to_images[anchor_class] if p != anchor_path])

        # 随机选一个不同类作为 negative
        neg_class = random.choice([c for c in self.all_classes if c != anchor_class])
        neg_path = random.choice(self.class_to_images[neg_class])

        # 加载图像
        anchor_img = self.loader(anchor_path)
        pos_img = self.loader(pos_path)
        neg_img = self.loader(neg_path)

        # 应用 transform
        anchor_img = self.transform(anchor_img)
        pos_img = self.transform(pos_img)
        neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

# 用于难例挖掘的数据集接口
def get_all_images(dataset, max_samples=None):
    """获取数据集中所有图像及其标签，用于难例挖掘"""
    from tqdm import tqdm
    import random
    
    all_paths = []
    all_labels = []
    
    # 首先收集所有路径和标签
    for class_idx in dataset.all_classes:
        for img_path in dataset.class_to_images[class_idx]:
            all_paths.append(img_path)
            all_labels.append(class_idx)
    
    # 如果指定了最大样本数，随机采样
    if max_samples and len(all_paths) > max_samples:
        print(f"Randomly sampling {max_samples} images from {len(all_paths)} total images")
        indices = random.sample(range(len(all_paths)), max_samples)
        all_paths = [all_paths[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
    
    # 加载和变换图像，显示进度
    print(f"Loading {len(all_paths)} images for hard mining...")
    all_images = []
    for img_path in tqdm(all_paths):
        img = dataset.loader(img_path)
        img = dataset.transform(img)
        all_images.append(img)
    
    return torch.stack(all_images), torch.tensor(all_labels), all_paths

# 测试数据集
class CowFaceTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.image_paths = [os.path.join(img_dir, img_name) for img_name in self.image_names]
        self.name_to_idx = {os.path.splitext(name)[0]: idx for idx, name in enumerate(self.image_names)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        name = os.path.splitext(self.image_names[idx])[0]
        return image, name  # 返回 embedding 时带上名字