'''针对时序序列数据（如传感器信号、音频序列、时间序列等）的数据增强 + 预处理工具集
提供模块化、可组合的序列变换操作（增强 / 预处理
通过随机噪声、缩放、拉伸、裁剪等增强手段丰富训练数据，提升模型泛化能力
统一数据格式（类型、维度、分布），适配深度学习模型的输入要求
'''
import numpy as np
import random
from scipy.signal import resample

#组合变换类 Compose（核心容器
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms #保存变换列表

    def __call__(self, seq):
        for t in self.transforms: #按顺序执行所有变换
            seq = t(seq)
        return seq

#维度转置
class Reshape(object):
    def __call__(self, seq):
        #打印(序列.形状)
        return seq.transpose() #调整维度
#非 numpy 类型转 numpy 数组
class to_numpy(object):
    def __call__(self, seq):
        #打印(seq.shape)
        return np.array(seq,dtype=np.float32)# 转为float32类型的numpy数组
#numpy 数组类型转换
class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)# 强制转为float32

#固定高斯噪声
class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma # 噪声标准差（控制噪声强度）

    def __call__(self, seq):
        # 生成与seq形状一致的高斯噪声（均值0），叠加到原序列
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

#随机高斯噪声
class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2): #50%概率跳过增强
            return seq
        else: #50%概率添加噪声
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

#固定幅度缩放
class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma #缩放因子的标准差

    def __call__(self, seq):
        # 生成每个通道的缩放因子（均值1，小幅波动）
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        # 构造与seq形状一致的缩放矩阵（每个通道所有时间步用同一因子）
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix #逐元素缩放

#随机缩放
class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2): #50%概率跳过
            return seq
        else: #50%概率缩放
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix

#拉伸/压缩增强时间维度
class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma #拉伸幅度（0.3到长度变化+-15%）

    def __call__(self, seq):
        if np.random.randint(2): #50%概率跳过
            return seq
        else:
            seq_aug = np.zeros(seq.shape) #初始化增强序列
            len = seq.shape[1] #原序列长度
            #计算新长度（随机拉伸/压缩）
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            # 遍历每个通道重采样
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)# 重采样到新长度
                if length < len: # 压缩：随机填充前/后段
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else: # 拉伸：随机截取前/后段
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug

#随机裁剪增强 RandomCrop（局部置 0）
class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len # 置0的连续长度

    def __call__(self, seq):
        if np.random.randint(2): # 50%概率跳过
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len # 避免越界
            random_index = np.random.randint(max_index)# 随机起始位置
            seq[:, random_index:random_index+self.crop_len] = 0# 局部置0
            return seq
#归一化类 Normalize（统一数据分布）
class Normalize(object):
    def __init__(self, type = "0-1"): # 支持3种归一化"0-1","-1-1","mean-std"
        self.type = type
    def __call__(self, seq):
        if  self.type == "0-1":
            #最小-最大归一化到[0,1]
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            # 最小-最大归一化到[-1,1]
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            #标准化（均值0，标准差1）
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq

#指定信噪比噪声
class AddNoiseSNR(object):
    """添加高斯噪声以达到指定信噪比（单位：分贝）。
    序列维度要求：多通道为（通道数，序列长度），单通道为（序列长度，）。
    """
    def __init__(self, snr):
        self.snr = snr #目标信噪比

    def __call__(self, seq):
        if self.snr is None:
            return seq
        arr = np.array(seq, dtype=np.float32)
        if arr.ndim == 1: #单通道扩展为（1，length）
            arr = arr[np.newaxis, :]
        # 计算各通道信号方差
        sig_var = np.var(arr, axis=1, keepdims=True)
        # 避免零方差
        sig_var = np.maximum(sig_var, 1e-12)
        # 生成原始高斯噪声
        noise = np.random.randn(*arr.shape).astype(np.float32)
        noise_std = np.std(noise, axis=1, keepdims=True)
        noise_std = np.maximum(noise_std, 1e-12)
        # 正确功率关系式：噪声功率 = 信号功率 / (10^(信噪比/10))
        target_noise_var = sig_var / np.power(10.0, (self.snr / 10.0))
        #调整噪声幅度以匹配目标方差
        noise = noise / noise_std * np.sqrt(target_noise_var)
        out = arr + noise
        #还原单通道维度
        if seq.ndim == 1:
            return out.squeeze(0)
        return out