import torch
nn = torch.nn
import numpy as np
import random
import os
import torch.nn.init as init
import math
import torch.nn.functional as F

seed = 999
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Gabor核的超参数范围
fmin = 0.03
fmax = 0.45

# 采用实虚部机制的传统卷积层
class BaseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
       #保存卷积层核心参数
        self.in_channels = in_channels#输入通道数
        self.out_channels = out_channels#输出通道数
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.stride = stride#步长
        self.padding = padding#填充

        self.phases = ['real', 'imag']#实虚部分支
       #初始化实/虚权重，形状为（2, out_channels, in_channels, kernel_size），第一维区分实部和虚部
        self.weight = torch.Tensor(len(self.phases), out_channels, in_channels, self.kernel_size)
        if bias:
            self.bias = torch.Tensor(len(self.phases), out_channels)#偏置：实部和虚部分别有一个偏置向量，形状为（2, out_channels）
        else:
            self.bias = None
        #逐个分支初始化权重和偏置，实部和虚部使用相同的初始化方法（Kaiming 均匀分布），但参数不同（实部和虚部权重独立，偏置独立）。如果 bias=False，则 self.bias 将保持为 None，不会被初始化
        for phase in self.phases:
            #权重初始化：Kaiming 均匀分布，适用于 ReLU 激活函数，能帮助缓解梯度消失问题。实部和虚部权重独立初始化，形状为（out_channels, in_channels, kernel_size）
            self.weight[self.phases.index(phase)] = torch.Tensor(out_channels, in_channels, self.kernel_size)
            init.kaiming_uniform_(self.weight[self.phases.index(phase)], a=math.sqrt(5))  # 初始权重
            if bias:
                #偏置初始化：根据输入通道数计算 fan_in，使用均匀分布在 [-1/sqrt(fan_in), 1/sqrt(fan_in)] 范围内初始化偏置。实部和虚部偏置独立初始化，形状为（out_channels,）
                self.bias[self.phases.index(phase)] = torch.Tensor(out_channels)
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[self.phases.index(phase)])  # 初始偏置
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias[self.phases.index(phase)], -bound, bound)
        # 若为BaseConv1d本身（非子类），将weight/bias转为可学习参数（nn.Parameter）
        if self.__class__.__name__ == 'BaseConv1d':
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)
    #前向传播
    def forward(self, input):
        result = {}
        # （实虚部分支分别做1D卷积）
        for phase in self.phases:
            if self.bias is None:
                result[phase] = F.conv1d(input, self.weight[self.phases.index(phase)],
                                         stride=self.stride, padding=self.padding)
            else:
                result[phase] = F.conv1d(input, self.weight[self.phases.index(phase)],
                                         bias=self.bias[self.phases.index(phase)],
                                         stride=self.stride, padding=self.padding)
        #卷积层的输出为实部与虚部的平方和的平方根（幅频包络，无相位信息）
        real_out = result[self.phases[0]]
        imag_out = result[self.phases[1]]

# 数值稳定的复数幅值包络
        mag_sq = (real_out.pow(2)+ imag_out.pow(2))

        output = torch.sqrt( mag_sq + 1e-8)
        return output

#参数化核函数的基类：继承BaseConv1d，为参数化核函数Gabor封装通用逻辑：参数范围、动态生成卷积核（代替固定权重）、保留梯度链路
#通过添加限制条件并重写前向传播函数，完成核函数的预处理
class BaseFuncConv1d(BaseConv1d):
    def __init__(self, *pargs, **kwargs):
       #过滤仅保留BaseFuncConv1d需要的参数（避免子类传参冲突）
        kwargs_new = {}
        for k in kwargs.keys():
            if k in ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']:
                kwargs_new[k] = kwargs[k]
        super().__init__(*pargs, **kwargs_new)#调用父类初始化
        # 若为BaseFuncConv1d本身，将weight/bias转为可学习参数
        if self.__class__.__name__ == 'BaseFuncConv1d':
            self.weight = torch.nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias)
            self.superparams = self.weight # 超参数别名（后续用于参数限制）

    def _clamp_parameters(self):
        with torch.no_grad():#无梯度更新（仅数值裁剪）
            for i in range(len(self.params_bound)):
                #按参数范围裁剪superparams（f0∈[0.03,0.45],sigma∈[0.1,1.0]
                self.superparams.data[:, :, i].clamp_(self.params_bound[i][0], self.params_bound[i][1])

    def WeightForward(self):
        if self.clamp_flag:
            self._clamp_parameters()#先裁剪参数
        l00 = []
        for phase in self.phases:
            l0 = []
            for i in range(self.superparams.shape[0]):
                l1 = []
                for j in range(self.superparams.shape[1]):
                    #调用子类实现的weightforward，生成单通道核
                    l1.append(self.weightforward(self.kernel_size, self.superparams[i, j], phase).unsqueeze(0))
                l0.append(torch.vstack(l1).unsqueeze(0))
            l00.append(torch.vstack(l0).unsqueeze(0))
        # 堆叠生成最终权重（保留梯度链路，而非普通Tensor）
        self.weight = torch.vstack(l00)

    def forward(self, input):
        if self.__class__.__name__ != 'BaseFuncConv1d':#子类调用时
            self.WeightForward()#动态生成卷积核
        return super().forward(input)#调用父类BaseConv1d的forward


#1D可学习gabor卷积层：基于BaseFuncConv1d实现1D Gabor卷积核，将卷积核从固定权重改为由可学习参数生成的Gabor函数，适配时域信号的频域特征提取
#初始化函数
class TFconv_Gabor(BaseFuncConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, clamp_flag=True,
                 params_bound=((fmin, fmax), (0.1, 1.0), (0.0, math.pi))):#超参数范围 (f0, sigma, theta)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag #是否开启参数裁剪
        self.params_bound = params_bound#超参数范围
        # superparams超参数: (out_channels, in_channels, N)映射到[f0, sigma, (theta)]
        self.superparams = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, len(params_bound))
        )
        self._reset_parameters()#初始化超参数
# =====================================================
# 保存Gabor初始参数
# 后续用于约束f0/sigma不要偏离初始有效滤波器组太远
# =====================================================
        self.register_buffer('superparams_init',self.superparams.detach().clone())        
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)
    #超参数初始化
    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            # 均匀初始化中心频率（覆盖fmin~fmax）
            f0_init = torch.linspace(fmin, fmax, shape.numel()).reshape(shape)
            self.superparams.data[:, :, 0] = f0_init
            # sigma 初始化为 0.5（适中带宽）
            self.superparams.data[:, :, 1].fill_(0.5)
            #  theta（第三个参数），随机初始化到 [0, pi]（1D下无作用）
            if self.superparams.shape[-1] > 2:
                self.superparams.data[:, :, 2].uniform_(0.0, math.pi)
            self.WeightForward()#初始化后生成初始核
    #Gabor核生成
    def weightforward(self, lens, params, phase):
        """
        params[0]: f0    (中心频率, normalized)
        params[1]: sigma (高斯标准差, 以半核长为单位归一化)
        params[2]: theta (可选，方向角/占位，1D 情况下不用于构建核，但仍作为可学习量保存)
        """
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())# 转为int（避免Tensor类型）
        half = max(lens // 2, 1)          # 半核长（half） 至少为 1，避免 kernel_size<=1 时除零
        #生成时间坐标：[-half,half)均匀分布
        T = torch.arange(-half, lens - half, dtype=torch.float32).to(params.device)
        # 高斯窗：exp(-t²/(2*(sigma*half)²)) 
        sigma = params[1].clamp(min=1e-3)#防止sigma为0
        denom = (sigma * half).clamp(min=1e-6)#分母保护
        gauss = torch.exp(-T.pow(2) / (2.0 * denom.pow(2)))
        #实部=高斯窗*余弦，虚部=高斯窗*正弦
        if self.phases.index(phase) == 0:
            result = gauss * torch.cos(2 * math.pi * params[0] * T)
        else:
            result = gauss * torch.sin(2 * math.pi * params[0] * T)
        return result

#处理1D原始信号/2D谱图，先将1D信号转为谱图（STFT），再用2D可学习Gabor核卷积，最后降维为1D特征，适配谱图的频域＋时域二维特征提取
#生成2D旋转网络坐标，基于可学习参数生成方向感知的2DGabor核，支持不同方向的频域特征提取
class TFconv_Gabor2D(nn.Module):
    """
    2D Gabor类似频谱图的预处理操作.
    输入可以是原始 1D 信号 (B,1,L) 或谱图 (B,1,F,T)。
    如果输入为 1D，会先用 torch.stft 计算谱图（幅度谱），然后对谱图应用多通道 2D Gabor 滤波器组。
    可学习超参 superparams: (out_channels, in_channels, 3) => [f0, sigma, theta]
      - f0: 归一化中心频率（范围 fmin~fmax）
      - sigma: 高斯窗宽度（相对于半核长，范围 0.1~1.0）
      - theta: 方向角（弧度），约束在 [0, pi]（即 0°~180°）
    输出: 对频率维度求平均后返回 (B, out_channels, T)，以保持与原 1D pipeline 接口兼容。
    """
    #初始化函数
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, clamp_flag=True,
                 params_bound=((fmin, fmax), (0.1, 1.0), (0.0, math.pi))):
        super().__init__()
        #保存核心参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else int(kernel_size[0])
        self.stride = stride
        self.padding = padding
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound

        # 可学习超参数superparams: (out_channels, in_channels, 3) 映射到 [f0, sigma, theta]
        self.superparams = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()#初始化超参数
        # 保存Gabor参数初始值，用于后续约束参数漂移
        self.register_buffer('superparams_init',self.superparams.detach().clone())
    #参数裁剪与初始化
    def _clamp_parameters(self):
        with torch.no_grad():
            for i in range(len(self.params_bound)):
                lo, hi = self.params_bound[i]
                self.superparams.data[:, :, i].clamp_(lo, hi)#裁剪超参数

    def _reset_parameters(self):
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            f0_init = torch.linspace(fmin, fmax, shape.numel()).reshape(shape)#f0均匀初始化
            self.superparams.data[:, :, 0] = f0_init
            self.superparams.data[:, :, 1].fill_(0.5)#sigma初始0.5
            # theta 随机初始化在 [0, pi]
            self.superparams.data[:, :, 2].uniform_(0.0, math.pi)
    #2D Gabor核生成
    def _build_kernel_2d(self, k, params, phase):
        half = max(k // 2, 1)
        coords = torch.linspace(-half, half, steps=k, device=params.device)#坐标范围[-half,half]
        Y, X = torch.meshgrid(coords, coords, indexing='ij')#生成2D网络坐标
        # 按theta旋转坐标（实现方向感知）
        theta = params[2]
        c = torch.cos(theta)
        s = torch.sin(theta)
        Xp = X * c + Y * s#旋转后的x坐标
        Yp = -X * s + Y * c#旋转后的Y坐标
        # 高斯窗（2D）：exp(-(Xp²+Yp²)/(2*(sigma*half)²))
        sigma = params[1].clamp(min=1e-3)
        denom = (sigma * half).clamp(min=1e-6)
        gauss = torch.exp(-(Xp.pow(2) + Yp.pow(2)) / (2.0 * denom.pow(2)))
        #实部=高斯窗*余弦，虚部=高斯窗*正弦（沿旋转后的Xp方向）
        if phase == 'real':
            kern = gauss * torch.cos(2 * math.pi * params[0] * Xp)
        else:
            kern = gauss * torch.sin(2 * math.pi * params[0] * Xp)
        return kern
    #前向传播函数
    def forward(self, input):
        # input: (B,1,L) or (B,1,F,T) or (B,C,F,T)
        if self.clamp_flag:
            self._clamp_parameters()#先裁剪超参数
        #步骤1：1D输入转谱图（STFT）
        if input.ndim == 3:
            # 通过torch.stft计算幅度频谱图
            x = input.squeeze(1)  # (B, L)
            n_fft = max(256, 2 ** int(math.ceil(math.log2(max(self.kernel_size * 4, 256)))))#保证足够频率分辨率
            hop = max(1, n_fft // 4)#75%重叠，提高时间分辨率
            # Hann窗避免频谱泄漏
            window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
            S_c = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
            #计算幅度谱：sqrt（实部²+虚部²）
            real = S_c.real
            imag = S_c.imag
            mag = torch.sqrt(real.pow(2) + imag.pow(2)) + 1e-12  # (B, F, T)数值保护
            # 对幅值谱做 log 压缩并按样本标准化，避免大幅值频带支配 2D Gabor 卷积
            mag = torch.log1p(mag)
            mean = mag.mean(dim=(-2, -1), keepdim=True)
            std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            mag = (mag - mean) / std
            spec = mag.unsqueeze(1)  # (B,1,F,T)
        elif input.ndim == 4:
            spec = input #直接使用输入谱图
        else:
            raise ValueError('Unsupported input ndim for TFconv_Gabor2D: {}'.format(input.ndim))
        #步骤2：构建2DGabor核（实/虚分支）
        B, C, Freq, T = spec.shape
        k = self.kernel_size
        pad = k // 2 #对称填充（保持尺寸）

        # 按每个输出/输入通道构建内核
        real_kernels = []
        imag_kernels = []
        for i in range(self.out_channels):
            rk_list = []
            ik_list = []
            for j in range(self.in_channels):
                params = self.superparams[i, j]
                kr = self._build_kernel_2d(k, params, 'real')#实部核
                ki = self._build_kernel_2d(k, params, 'imag')#虚部核
                rk_list.append(kr.unsqueeze(0))
                ik_list.append(ki.unsqueeze(0))
            # 堆叠输入通道 (in, k, k)
            real_kernels.append(torch.vstack(rk_list).unsqueeze(0))
            imag_kernels.append(torch.vstack(ik_list).unsqueeze(0))

        # 堆叠输出通道(out, in, k, k)
        real_kernels = torch.vstack(real_kernels).to(spec.device)
        imag_kernels = torch.vstack(imag_kernels).to(spec.device)

        #步骤3：2D卷积（实/虚分支）
        real_out = F.conv2d(spec, real_kernels, bias=None, stride=1, padding=pad)
        imag_out = F.conv2d(spec, imag_kernels, bias=None, stride=1, padding=pad)
        mag_out = torch.sqrt(real_out.pow(2) + imag_out.pow(2) + 1e-12)#幅度包络

        # 步骤4：频率维自适应能量加权池化。
        # 相比简单 mean，可避免强响应被大量无关频点稀释。
        freq_score = mag_out.mean(dim=-1)  # (B, out_channels, F)
        freq_weight = torch.softmax(freq_score, dim=-1)
        out = (mag_out * freq_weight.unsqueeze(-1)).sum(dim=2)  # (B, out_channels, T)

        # 不再插值回 2048 个原始采样点。后续 1D CNN 含 AdaptiveMaxPool，可直接处理 STFT 帧序列。
        return out


if __name__ == '__main__':
    # test TFconv_Gabor and TFconv_Gabor2D
    for item in [TFconv_Gabor, TFconv_Gabor2D]:
        print(item.__name__)
        # TFconv_Gabor2D接收整数类型的卷积核尺寸参数，并将在内部计算频谱图
        #初始化模型：1输入通道，8输出通道，核尺寸15，填充7（保持尺寸）
        model = item(1, 8, 15, padding=7, stride=1, bias=False, clamp_flag=True)
        input = torch.randn([2, 1, 1024])
        out = model(input)#前向传播
        out.mean().backward()#反向传播（测试梯度回传）
        print(out.shape)
        print(f'{item.__name__:s} test pass')
