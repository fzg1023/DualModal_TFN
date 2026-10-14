'''对比可学习时频卷积与固定时频卷积双模态网络性能
实验组（learnable）：时频卷积（funconv）的超参数随训练更新
对照组（fixed）：冻结时频卷积超参数（仅其他层可训练）
通过 “相同初始化 + 单一变量 + 分层抽样” 的对比实验，验证可学习时频卷积对模型性能的影响。
终端python tests/compare_fixed_vs_learnable.py --data_dir "D:\Desktop\BJTU_RAO_Bogie_Datasets"
'''
import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

#添加仓库根目录
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Models.TFN import DualModal_TFN
from tools.frequency_analysis import find_csv_files, load_signals_from_csv

# 最小数据集：将窗口信号分帧，并为两种模态复用同一信号
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, signals, window=1024, hop=512, max_samples_per_file=10):
        self.items = []
        self.window = window
        for sig, label in signals:
            L = len(sig)
            starts = list(range(0, max(1, L - window + 1), hop))
            if len(starts) > max_samples_per_file:
                starts = random.sample(starts, max_samples_per_file)
            for s in starts:
                windowed = sig[s:s+window]
                if len(windowed) < window:
                    pad = np.zeros(window, dtype=np.float32)
                    pad[:len(windowed)] = windowed
                    windowed = pad
                self.items.append((windowed.astype(np.float32), label))
    #返回数据集总样本数
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        # 本次快速测试中，振动控制算法与蚁群算法返回结果一致
        xb = torch.from_numpy(x).unsqueeze(0)  # (1, L)
        return xb, xb.clone(), int(y)


def build_signal_list(data_dir, max_files=8): #最大加载文件数8
    files = find_csv_files(data_dir, max_files=max_files)
    sigs = load_signals_from_csv(files)
    #0=健康, 1=内圈故障, 2=外圈故障, 3=滚珠故障, 4=保持架故障
    labeled = []
    for f, s in zip(files, sigs):
        fl = f.lower()
        # 兼容两种目录命名：「编号前缀」（如1_healthy）和「关键词命名」（如inner_race）
        if '_la0_' in fl or '/1_' in fl:
            lbl = 0   # 健康样本
        elif '_la1_' in fl or '/2_' in fl:
            lbl = 1   # 内圈故障
        elif '_la2_' in fl or '/3_' in fl:
            lbl = 2   # 外圈故障
        elif '_la3_' in fl or '/4_' in fl:
            lbl = 3   # 滚珠（滚动体）故障
        elif '_la4_' in fl or '/5_' in fl:
            lbl = 4   # 保持架故障
        else:
            # 未知类别自动跳过，避免异常标签干扰实验
            print(f"Skip unknown category file: {f}")
            continue
        labeled.append((s, lbl))
    return labeled

#训练函数
def train_one_epoch(model, device, loader, opt, criterion):
    model.train()
    total = 0
    correct = 0
    for vib, aco, y in loader:
        vib = vib.to(device)
        aco = aco.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(vib, aco)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()
    return correct/total if total>0 else 0.0

#评估函数
def eval_model(model, device, loader, criterion):
    model.eval()
    total = 0 
    correct = 0
    with torch.no_grad():
        for vib, aco, y in loader:
            vib = vib.to(device)
            aco = aco.to(device)
            y = y.to(device)
            logits = model(vib, aco)
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return correct/total if total>0 else 0.0

#对比函数（核心）：构建数据集、初始化对比模型、保证初始化一致、冻结固定模型参数、训练＋评估、输出准确率/参数对比
def compare(data_dir):
    device = torch.device('cpu')
    signals = build_signal_list(data_dir, max_files=20)
    if len(signals) < 2:
        print('Not enough signals found in', data_dir)
        return
# 先全量窗口化，再对样本分层 
    all_ds = WindowDataset(signals, window=1024, hop=512, max_samples_per_file=6)
    # 提取所有样本和标签
    all_x = []
    all_y = []
    for x_vib, x_aco, y in all_ds:
        all_x.append((x_vib, x_aco))
        all_y.append(y)
    
    # 对窗口样本做分层抽样
    train_idx, val_idx = train_test_split(
        list(range(len(all_x))), 
        test_size=0.3, 
        random_state=42, 
        stratify=all_y
    )
    train_samples = [all_x[i] + (all_y[i],) for i in train_idx]
    val_samples = [all_x[i] + (all_y[i],) for i in val_idx]

    # 构造简单的样本列表数据集（适配DataLoader）
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]
    
    train_ds = SimpleDataset(train_samples)
    val_ds = SimpleDataset(val_samples)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)

    # 基准模型：可训练时域卷积（默认配置）
    model_learn = DualModal_TFN(in_channels=1, out_channels=5, mid_channel=8, clamp_flag=True).to(device)
    # 控制项：冻结泛化卷积超参数（滤波器固定）
    model_fixed = DualModal_TFN(in_channels=1, out_channels=5, mid_channel=8, clamp_flag=True).to(device)
    # 复制初始状态以保证初始化操作一致
    model_fixed.load_state_dict(model_learn.state_dict())
    # 冻结融合卷积模块参数
    for branch in [model_fixed.vib_branch, model_fixed.aco_branch]:
        if hasattr(branch, 'funconv') and hasattr(branch.funconv, 'superparams'):
            branch.funconv.superparams.requires_grad = False

    # 优化器：自动排除无梯度参数
    opt_learn = torch.optim.Adam(filter(lambda p: p.requires_grad, model_learn.parameters()), lr=1e-3)
    opt_fixed = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fixed.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 快速训练：10轮迭代
    epochs = 10
    print('Train samples:', len(train_ds), 'Val samples:', len(val_ds))
    for e in range(epochs):
        acc_learn = train_one_epoch(model_learn, device, train_loader, opt_learn, criterion)
        acc_fixed = train_one_epoch(model_fixed, device, train_loader, opt_fixed, criterion)
        val_learn = eval_model(model_learn, device, val_loader, criterion)
        val_fixed = eval_model(model_fixed, device, val_loader, criterion)
        print(f'Epoch {e+1}/{epochs} | learn_train={acc_learn:.4f} fixed_train={acc_fixed:.4f} | learn_val={val_learn:.4f} fixed_val={val_fixed:.4f}')

    # 可学习模型超参数调整的最终对比
    sp_learn = model_learn.vib_branch.funconv.superparams.detach().cpu().numpy()
    sp_fixed = model_fixed.vib_branch.funconv.superparams.detach().cpu().numpy()
    print('learnable superparams sample (first out,in):', sp_learn[0,0])
    print('fixed superparams sample (first out,in):', sp_fixed[0,0])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    args = p.parse_args()
    compare(args.data_dir)
