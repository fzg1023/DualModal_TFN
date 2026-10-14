
from torch.utils.data import Dataset


class dataset(Dataset):
    #初始化数据存储（序列 + 标签）和变换函数，完成数据集的 “加载” 准备
    def __init__(self, list_data, transform=None):
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        self.transforms = transform

    #提供数据集总长度，支撑DataLoader的批量计算、采样等逻辑
    def __len__(self):
        return len(self.seq_data)
    #按需加载单个样本（懒加载），支持数据变换，是DataLoader获取数据的核心
    def __getitem__(self, item):
        seq = self.seq_data[item]
        label = self.labels[item]
        if self.transforms:
           seq = self.transforms(seq)
        return seq, label