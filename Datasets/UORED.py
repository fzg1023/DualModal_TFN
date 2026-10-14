import pandas as pd
from Datasets.Dataset_utils.DatasetsBase import dataset
import Datasets.Dataset_utils.sequence_aug as aug1d
from Datasets.get_files.generalfunciton import data_transforms1d, balance_label, add_noise
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
import pickle

seed = 999
np.random.seed(seed)
random.seed(seed)


class UORED(object):
    """
    该代码实现会加载渥太华大学（UORED-VAFCLS）数据集的 CSV 原始文件，并生成双模态样本（加速度计数据、声学数据），样本维度为 (2, L).
    """
    # 5类（健康 内圈 外圈 滚珠 保持架）
    num_classes = 5
    inputchannel = 1

    def __init__(self, args):
        self.args = args

    def _scan_csv_files(self, root):
        """扫描CSV文件夹并返回由（文件路径，标签）组成的列表"""
        class_map = {
            '1_Healthy': 0,
            '2_Inner_Race_Faults': 1,
            '3_Outer_Race_Faults': 2,
            '4_Ball_Faults': 3,
            '5_Cage_Faults': 4,
        }
        csv_root = os.path.join(root, '1_CSV_Raw_Data_Files (.csv)')
        files = []
        labels = []
        if not os.path.isdir(csv_root):
            raise FileNotFoundError(f'UORED CSV folder not found: {csv_root}')
        for cls_dir, lab in class_map.items():
            dir_path = os.path.join(csv_root, cls_dir)
            if not os.path.isdir(dir_path):
                continue
            for fn in os.listdir(dir_path):
                if fn.lower().endswith('.csv'):
                    files.append(os.path.join(dir_path, fn))
                    labels.append(lab)
        return files, labels

    def _load_csv_sample(self, filepath, signal_size, step, downsample_rate=1, SNR=None):
        """读取CSV文件并切分为窗口数据，返回由多个形状为(2,L)的数组组成的列表."""
        df = pd.read_csv(filepath)
        # 预期列：加速度计、声学传感器
        if 'Accelerometer' not in df.columns or 'Acoustic' not in df.columns:
            raise ValueError(f'CSV missing required columns: {filepath}')
       #将加速度/声学列转为float32数组，并根据downsample_rate进行下采样（默认不下采样）。注意：下采样率必须为正整数，否则将被视为1（即不下采样）。
        acc = df['Accelerometer'].to_numpy(dtype=np.float32)[::max(int(downsample_rate),1)]
        aco = df['Acoustic'].to_numpy(dtype=np.float32)[::max(int(downsample_rate),1)]
        # 注意：请勿在此处添加信噪比噪声；此处将实时应用基于信噪比的数据增强方法
        # 在变换流程中，使噪声能够随训练轮次发生变化。
        data = []
        start = 0
        end = signal_size
        while end <= acc.shape[0] and end <= aco.shape[0]:
            v = acc[start:end]
            a = aco[start:end]
            sample = np.stack([v, a], axis=0)  # shape (2, L)
            data.append(sample)
            start += step
            end += step
        return data
    #预加载/缓存数据集
    def _preload(self, prefile_dir, args):
        """预加载或构建数据集，并以 pickle 格式缓存"""
        if os.path.exists(prefile_dir):
            try:
                with open(prefile_dir, 'rb') as f:
                    data_pd, storage_args, label_name = pickle.load(f)
                    if storage_args == args:
                        return data_pd, label_name
            except Exception:
                pass
        #缓存失效则调用_scan_csv_files获取文件列表
        files, file_labels = self._scan_csv_files(args['root'])
        all_data = []
        all_label = []
        signal_size = args['signal_size']
        down = args.get('downsample_rate', 1)
        SNR = args.get('SNR', None)
        step = int(signal_size // max(int(down),1))
        for fp, lab in zip(files, file_labels):
            samples = self._load_csv_sample(fp, signal_size, step, downsample_rate=down, SNR=SNR)
            all_data.extend(samples)
            all_label.extend([lab] * len(samples))

        label_name = ['Healthy', 'Inner', 'Outer', 'Ball', 'Cage']
        data_pd = pd.DataFrame({"data": all_data, "label": all_label})
        try:
            with open(prefile_dir, 'wb') as f:
                pickle.dump((data_pd, args, label_name), f)
        except Exception:
            pass
        return data_pd, label_name
    #核心预处理，拆分训练/测试/验证集，并应用数据增强
    def data_preprare(self, signal_size=1024, SNR_train=None, SNR_val=None):
        split_seed = int(self.args.get('split_seed', 999))
        # 基于分组拆分：将样本按group_size（默认值为 100）划分为多个组，
        # 然后对每个分组，将前 80% 划分为训练集，接下来的 10% 划分为测试集，最后 10% 划分为验证集。
        group_size = int(self.args.get('group_size', 100))
        temp_args = {'root': self.args['data_dir'], 'signal_size': signal_size,
                 'downsample_rate': 1}
        data_pd, label_name = self._preload(os.path.join(self.args['data_dir'], 'data_buffer_uored.pkl'), temp_args)
        # 平衡每个类别的样本（可选目标设为最小类别数量）
        data_pd = balance_label(data_pd, 'label', None)

        # 确定性地打乱顺序
        data_pd = data_pd.sample(frac=1, random_state=split_seed).reset_index(drop=True)

        # 将数据分块编组，并按80/10/10的比例拆分每组数据（含取整处理）
        n = len(data_pd)
        groups = [data_pd.iloc[i:i + group_size] for i in range(0, n, group_size)]
        train_parts = []
        test_parts = []
        val_parts = []
        for g in groups:
            L = len(g)
            if L == 0:
                continue
            # 将每组最后 20% 的数据作为测试集，每组前 80% 的数据用于训练和验证
            n_trainval = int(np.floor(L * 0.8))
            n_test = L - n_trainval
            if n_trainval <= 0:
                # fallback: everything goes to test
                if n_test > 0:
                    test_parts.append(g)
                continue
            # 将前80%的数据划分为训练集和验证集（训练验证集内部按照9:1的比例拆分）
            n_train = int(np.floor(n_trainval * 0.9))
            n_val = n_trainval - n_train
            if n_train > 0:
                train_parts.append(g.iloc[:n_train])
            if n_val > 0:
                val_parts.append(g.iloc[n_train:n_train + n_val])
            if n_test > 0:
                test_parts.append(g.iloc[n_trainval:])
       # 将所有分块的训练/测试/验证数据合并成完整的训练/测试/验证 DataFrame
        train_pd = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=data_pd.columns)
        test_pd = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=data_pd.columns)
        val_pd = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=data_pd.columns)

        # v5: expose raw split metadata so DualModal_TFN can warm-start its INTERNAL
        # spectral submodule from TRAIN only. This does not alter returned datasets.
        self.raw_splits = {
            "train": train_pd[["data", "label"]].copy(),
            "val": val_pd[["data", "label"]].copy(),
            "test": test_pd[["data", "label"]].copy(),
        }

        #将信噪比参数传入训练、验证、测试的数据变换流程，以实现实时噪声增强
        # 训练集使用训练信噪比；验证集/测试集使用验证信噪比（默认与训练信噪比一致）
        train_dataset = dataset(list_data=train_pd, transform=data_transforms1d(aug1d, 'train', self.args.get('normlizetype', 'mean-std'), snr=SNR_train))
        test_dataset = dataset(list_data=test_pd, transform=data_transforms1d(aug1d, 'test', self.args.get('normlizetype', 'mean-std'), snr=SNR_val))
        val_dataset = dataset(list_data=val_pd, transform=data_transforms1d(aug1d, 'val', self.args.get('normlizetype', 'mean-std'), snr=SNR_val))

        # 保持向后兼容的返回顺序：历史沿用顺序为(训练集, 验证集)。
        # 新逻辑会按此顺序依次返回训练集、验证集、测试集。
        return (train_dataset, val_dataset, test_dataset), label_name



if __name__ == '__main__': # 检查文件运行正常
    args = {'data_type': 'time', 'data_dir': './Datasets_dir/UORED','normlizetype': 'mean-std'}
    uored = UORED(args)
    out = uored.data_preprare()
    print(1)
