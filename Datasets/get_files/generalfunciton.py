'''面向一维信号处理的通用工具库
向信号添加指定信噪比（SNR）的高斯噪声
构建区分训练 / 验证 / 测试集的一维信号增强流水线
实现信号的时域、频域（FFT）、时频域（多种 STFT 变体）变换，将一维信号转为二维时频图
对带标签的数据集进行标签平衡（欠采样），解决类别不平衡问题
保证实验可复现性（固定随机种子
'''
import numpy as np
# import pywt
from scipy import signal
import pandas as pd
from torch.utils.data import Dataset
import random

# random.seed(999)
seed = 999
np.random.seed(seed)
random.seed(seed)

def add_noise(sig,SNR): # 向信号添加噪声
    noise = np.random.randn(*sig.shape)
    noise_var = sig.var() / np.power(10,(SNR/20))
    noise = noise /noise.std() * np.sqrt(noise_var)
    return sig + noise


def data_transforms1d(aug, dataset_type="train", normlize_type="1-1", aug_flag=False, snr=None):
    """构建变换流程。
    若传入`snr`参数，则对训练集、验证集、测试集实时执行基于信噪比的噪声增强，保证评估所用信噪比与设定值一致。
    训练阶段默认开启额外随机数据增强操作。
    """
    # base transforms shared
    base_trans = [aug.to_numpy(), aug.Normalize(normlize_type)]
    # 训练阶段默认开启额外数据增强
    if aug_flag or dataset_type == 'train':
        extra = [aug.RandomAddGaussian(), aug.RandomScale(), aug.RandomStretch(), aug.RandomCrop()]
    else:
        extra = []
    # 如果提供了 snr，则对训练/验证/测试都应用 SNR 噪声变换（默认验证/测试与训练使用相同 SNR）
    if snr is not None:
        train_trans = aug.Compose(base_trans + extra + [aug.AddNoiseSNR(snr), aug.Retype()])
        val_trans = aug.Compose([aug.to_numpy(), aug.Normalize(normlize_type), aug.AddNoiseSNR(snr), aug.Retype()])
        test_trans = aug.Compose([aug.to_numpy(), aug.Normalize(normlize_type), aug.AddNoiseSNR(snr), aug.Retype()])
    else:
        train_trans = aug.Compose(base_trans + extra + [aug.Retype()])
        val_trans = aug.Compose([aug.to_numpy(), aug.Normalize(normlize_type), aug.Retype()])
        test_trans = val_trans #测试集复用验证集变换
    transforms = {'train': train_trans, 'val': val_trans, 'test': test_trans}
    return transforms[dataset_type] #返回对应数据集的变换流水线

class sig_process(object):
    nperseg = 30 # STFT的默认分段长度（每段信号的点数）
    adjust_flag = False# STFT是否调整到输出接近正方形的标记
    def __init__(self):
        super(sig_process,self).__init__()
    #时域变换
    @classmethod
    def time(cls,x):
        return x #直接返回原始信号
    #FFT频域变换
    @classmethod
    def fft(cls,x):
        x = x - np.mean(x)# 去除直流分量（均值），避免直流分量影响频域分析
        x = np.fft.fft(x)# 快速傅里叶变换，得到复数形式的频域信号
        x = np.abs(x) / len(x)# 计算幅值谱并归一化（除以信号长度）
        x = x[range(int(x.shape[0] / 2))]# 取正频率部分（FFT结果对称，仅保留前半）
        x[1:-1] = 2*x[1:-1]# 修正幅值（除直流和最高频外，幅值需×2
        return x
    #切片重塑
    @classmethod
    def slice(cls,x):
        w = int(np.sqrt(len(x)))# 计算信号长度的平方根（取整）
        img = x[:w**2].reshape(w,w)# 截取前w²个点，重塑为w×w的正方形二维数组（模拟图像
        return img
    #自适应STFT
    @classmethod
    def STFT(cls,x,verbose=False):
        while not cls.adjust_flag:# 循环调整nperseg，直到输出时频图接近正方形
            _,_, Zxx = signal.stft(x, nperseg=cls.nperseg)# 计算STFT，得到复数时频矩阵
            if abs(Zxx.shape[0] - Zxx.shape[1]) < 2:# 行列数差值<2，视为接近正方形
                cls.adjust_flag = True# 停止调整
            elif Zxx.shape[0] > Zxx.shape[1]:# 行数>列数，减小分段长度
                cls.nperseg -= 1
            else:# 列数>行数，增大分段长度
                cls.nperseg += 1
        f, t, Zxx = signal.stft(x, nperseg=cls.nperseg)# 用调整后的nperseg重新计算STFT
        img = np.abs(Zxx) / len(Zxx)# 计算幅值时频图并归一化
        if verbose:
            return f, t, img# 详细模式返回频率轴、时间轴、时频图
        else:
            return img# 仅返回时频图
    #固定参数STFT变体
    @classmethod
    def STFT8(cls,x,Nc=8):
        f, t, Zxx = signal.stft(x, nperseg=Nc*2-1,noverlap=Nc*2-2)# 固定分段长度和重叠数
        img = np.abs(Zxx) / len(Zxx)# 幅值归一化
        return img
    @classmethod
    def STFT16(cls,x):
        return sig_process.STFT8(x,Nc=16)# Nc=16的固定参数STFT

    @classmethod
    def STFT32(cls,x):
        return sig_process.STFT8(x, Nc=32)# Nc=32

    @classmethod
    def STFT64(cls,x):
        return sig_process.STFT8(x, Nc=64)

    @classmethod
    def STFT128(cls,x):
        return sig_process.STFT8(x, Nc=128)
    #自定义参数STFT
    @classmethod
    def mySTFT(cls,x,verbose=False,nperseg=256,noverlap = None):
        if not noverlap:
            noverlap = nperseg//2# 未指定重叠数时，默认取分段长度的1/2
        f, t, Zxx = signal.stft(x, nperseg=nperseg,noverlap=noverlap)# 自定义分段长度和重叠数
        img = np.abs(Zxx) / len(Zxx)# 幅值归一化
        if verbose:
            return f, t, img# 详细模式返回频率/时间轴+时频图
        else:
            return img# 仅返回时频图

def balance_label(df,labelcol = 'label', number=None):
    #处理空数据框
    if df is None or len(df) == 0:
        return df

    value_count = df[labelcol].value_counts()# 统计每个标签的样本数
    # 若未找到标签，则返回原文
    if value_count.size == 0:
        return df

    if number is None:
        number = int(value_count.to_numpy().min())

    # 若计算得出的数值小于等于零，则返回原始数值
    if number <= 0:
        return df

    parts = []
    for lab in value_count.index.sort_values().to_list():# 遍历所有标签（排序）
        idxs = df.index[df[labelcol] == lab].to_list()# 获取该标签的所有样本索引
        parts.append(df.loc[idxs[:number]])# 截取前number个样本，加入列表
    new_df = pd.concat(parts, ignore_index=True)# 拼接所有标签的采样结果，重置索引
    return new_df