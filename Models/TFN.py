from Models.TFconvlayer import *
from Models.BackboneCNN import CNN
from torch.nn import Conv1d
from utils.mysummary import summary
import torch.nn.functional as F
import math

#TFN的基类（TFN的核心框架）
class Base_FUNC_CNN(CNN):
    FuncConv1d = BaseFuncConv1d #默认预处理层（时频卷积基类）
    funckernel_size = 21 #默认预处理层核大小

    def __init__(self, in_channels=1, out_channels=10, kernel_size=15, clamp_flag=True, mid_channel=16):
        super().__init__(in_channels, out_channels, kernel_size) #调用父类CNN初始化
        #通过修改输入通道数重新初始化第一层网络（适配预处理层的输出通道mid_channel)
        args = {x: getattr(self.layer1[0], x) for x in
                ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']}
        args['bias'] = None if (args['bias'] is None) else True
        args['in_channels'] = mid_channel #layer1输入=预处理层输出
        self.layer1[0] = nn.Conv1d(**args)
        # 将TF卷积层用作首个预处理层
        self.funconv = self.FuncConv1d(in_channels, mid_channel, self.funckernel_size,
                                       padding=self.funckernel_size // 2,
                                       bias=False, clamp_flag=clamp_flag)
        self.superparams = self.funconv.superparams #保存预处理层超参数

    def forward(self, x):
        x = self.funconv(x) #步骤一：时频预处理（核心差异点）
        return super().forward(x) #步骤二：基础CNN前向传播
    #提取预处理层权重/超参数（用于可视化分析）
    def getweight(self):
        weight = self.funconv.weight.cpu().detach().numpy()
        superparams = self.funconv.superparams.cpu().detach().numpy()
        return weight, superparams

#随机卷积基线：传统随机权重Conv1d（无参数学习），用于对比验证Gabor时频滤波的有效性
class Random_conv(Conv1d):
    def __init__(self, *pargs, **kwargs):
        new_kwargs = {k:v for k,v in kwargs.items() if k in ['in_channels','out_channels','kernel_size','stride','padding','bias']}
        super().__init__(*pargs, **new_kwargs)
        self.superparams = self.weight
#基线模型:预处理层替换为随机卷积，以随机卷积层作为首层预处理层的骨干卷积神经网络
class Random_CNN(Base_FUNC_CNN):
    FuncConv1d = Random_conv #重写预处理层为随机卷积
    def __init__(self, mid_channel=16, **kwargs):
        self.funckernel_size = mid_channel * 2 - 1 #核大小随滤波器数量调整
        super().__init__(mid_channel=mid_channel, **kwargs)


# Gabor 单模态 TFN
class TFN_Gabor(Base_FUNC_CNN):
    """
    TFN 第一层替换为参数化 Gabor 滤波器组
    超参数: f0 (中心频率), sigma (高斯宽度)
    kernel_size = mid_channel * 2 - 1
    """
    FuncConv1d = TFconv_Gabor #重写预处理层为Gabor时频卷积

    def __init__(self, mid_channel=64,  gabor1d_kernel_size=63,**kwargs):
        self.funckernel_size =int(gabor1d_kernel_size)
        # 保证核长度为奇数
        if self.funckernel_size % 2 == 0:
            self.funckernel_size += 1
        super().__init__(mid_channel=mid_channel, **kwargs)


# 注意力模块（Channel + Spatial — CBAM 简化版，用于特征融合）
class ChannelAttention1d(nn.Module):
    """SE-style 通道注意力：自适应加权不同通道特征"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)#降维后的通道数（至少8，避免维度过小）
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),#降维
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),#升维
            nn.Sigmoid(),#输出0-1权重
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出：加权后的（B，C）
        return x * self.fc(x)

#特征融合门控：振动+声音特征加权融合
class FusionGate(nn.Module):
    """
    特征级自适应融合门控：将两路特征拼接后，通过 SE 加权再相加
    输入: feat_v (B, D),  feat_a (B, D)
    输出: fused (B, D)
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.ca = ChannelAttention1d(feat_dim * 2, reduction=4)#输入为两路拼接（2D）
        self.proj = nn.Linear(feat_dim * 2, feat_dim)#投影回原维度

    def forward(self, feat_v: torch.Tensor, feat_a: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([feat_v, feat_a], dim=-1)   # (B, 2D)拼接两路特征
        cat = self.ca(cat)                            #通道注意力加权（自适应调整权重）
        fused = self.proj(cat)                        # (B, D)投影回原维度
        return fused


# 双模态特征提取骨干（不含分类头，供双模态分支复用）
class _TFN_Backbone(nn.Module):
    """
    内部辅助类：将 Base_FUNC_CNN 拆成"特征提取"与"分类头"两部分，
    供 DualModal_TFN 的每条分支使用。
    FuncConv1d 默认 TFconv_Gabor；也可传其他 TFconv_* 类。
    输出 forward() 返回 (B, 64) 的特征向量（去掉最后 fc 层）。
    """
    # DualModal 默认回到 1D Gabor：对原始时域信号直接滤波，
    # 避免强噪声下 STFT+2D 卷积对随机谱纹理过拟合，同时显著降低计算量。
    # 2D Gabor 功能仍保留，可由 DualModal_TFN(gabor_mode='2d') 显式启用。
    _default_FuncConv1d = TFconv_Gabor

    def __init__(self, in_channels=1, mid_channel=16, kernel_size=15,
                 clamp_flag=True, func_conv_cls=None, gabor2d_kernel_size=15,
                 gabor1d_kernel_size=63): 
        super().__init__()
        _func = func_conv_cls if func_conv_cls is not None else self._default_FuncConv1d
        #初始化Gabor预处理层
        # 1D Gabor/Random baseline may keep k=2*C-1; 2D Gabor must use a small spatial kernel.
        # A 127x127 kernel on a ~65 x tens-of-frames spectrogram is numerically and physically inappropriate.
        if _func is TFconv_Gabor2D:
            funckernel_size = int(gabor2d_kernel_size)
            if funckernel_size % 2 == 0:
                funckernel_size += 1
        elif _func is TFconv_Gabor:
            # 固定 63 点核，不再随 mid_channel 扩成 127/255 点，速度更稳定。
            funckernel_size = int(gabor1d_kernel_size)
            if funckernel_size % 2 == 0:
                funckernel_size += 1
        else:
            funckernel_size = mid_channel * 2 - 1
        self.funconv = _func(
            in_channels, mid_channel, funckernel_size,
            padding=funckernel_size // 2, bias=False, clamp_flag=clamp_flag
        )
        self.superparams = self.funconv.superparams
        #特征提取层（无分类头，输出64维特征）
        # layer1 ~ layer5（与 CNN 一致，但 in_channels 接 mid_channel）
        self.layer1 = nn.Sequential(
            nn.Conv1d(mid_channel, 16, kernel_size=kernel_size, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))
        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.squeeze(1) #适配4维输入（B,1,C,L）→ 3维（B,C,L）
        x = self.funconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1) #展平：(B,128,4) → (B,512)
        x = self.layer5(x)   # 输出：(B, 64)特征向量
        return x


# v5: DualModal_TFN 内部的轻量双模态频谱分支。
# 该分支完全属于 PyTorch 模型本体，不依赖 sklearn / SVM / 外部 teacher。
class _DualSpectralHead(nn.Module):
    """Learnable two-modal spectral classifier used inside DualModal_TFN.

    Features per modality:
      - pooled log FFT magnitude (``bins``)
      - 24 low-order cepstral coefficients
      - 24 short-lag normalized autocorrelation coefficients

    The input feature mean/std are train-split-only buffers populated once by
    ``train_utils``.  If pretraining is disabled, zero/one defaults remain valid.
    """
    def __init__(self, out_channels: int, bins: int = 128, hidden: int = 128):
        super().__init__()
        self.bins = int(max(32, bins))
        self.cep_dim = 24
        self.ac_dim = 24
        self.per_modal_dim = self.bins + self.cep_dim + self.ac_dim
        self.feature_dim = self.per_modal_dim * 2
        self.register_buffer('feature_mean', torch.zeros(self.feature_dim))
        self.register_buffer('feature_std', torch.ones(self.feature_dim))
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(p=0.08),
            nn.Linear(int(hidden), out_channels),
        )

    @staticmethod
    def _standardize_wave(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, 0, :]
        x = x - x.mean(dim=-1, keepdim=True)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return x / rms

    def _features_one(self, x: torch.Tensor) -> torch.Tensor:
        z = self._standardize_wave(x)
        mag = torch.fft.rfft(z, dim=-1).abs()[..., 1:]
        pooled = F.adaptive_avg_pool1d(mag.unsqueeze(1), self.bins).squeeze(1)
        log_spec = torch.log1p(pooled)

        log_mag = torch.log1p(mag)
        cep = torch.fft.rfft(log_mag, dim=-1).real[..., :self.cep_dim]
        ac = torch.fft.irfft(mag * mag, dim=-1).real[..., :self.ac_dim]
        ac = ac / ac[..., :1].abs().clamp_min(1e-6)
        return torch.cat([log_spec, cep, ac], dim=-1)

    def extract_features(self, x_vib: torch.Tensor, x_aco: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._features_one(x_vib), self._features_one(x_aco)], dim=-1)

    @torch.no_grad()
    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.reshape(-1).to(device=self.feature_mean.device, dtype=self.feature_mean.dtype)
        std = std.reshape(-1).to(device=self.feature_std.device, dtype=self.feature_std.dtype)
        if mean.numel() != self.feature_dim or std.numel() != self.feature_dim:
            raise ValueError(
                f'spectral feature stats have wrong size: mean={mean.numel()}, std={std.numel()}, '
                f'expected={self.feature_dim}'
            )
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-5))

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        z = (features - self.feature_mean) / self.feature_std.clamp_min(1e-5)
        return self.net(z)

    def forward(self, x_vib: torch.Tensor, x_aco: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.extract_features(x_vib, x_aco))


# 双模态 TFN（v5：原 TFN 双模态主干 + 模型内部可学习频谱分支）
class DualModal_TFN(nn.Module):
    """
    双模态故障诊断网络。

    v5 保留原 DualModal_TFN 的双分支 TFN 与特征/决策融合主体，
    并在模型内部加入一个轻量、可学习的双模态频谱分类分支。
    频谱分支只使用训练集标签进行一次 warm-start，随后与 TFN 端到端联合训练；
    验证/测试不依赖 SVM、teacher 或 sklearn 推理。
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 10,
                 kernel_size: int = 15,
                 mid_channel: int = 16,
                 clamp_flag: bool = True,
                 use_decision_fusion: bool = False,
                 decision_alpha: float = 0.3,
                 FuncConv1d_cls=None,
                 gabor2d_kernel_size: int = 15,
                 gabor1d_kernel_size: int = 63,
                 gabor_mode: str = '2d',
                 use_internal_spectral: bool = True,
                 internal_spectral_bins: int = 128,
                 internal_spectral_hidden: int = 128,
                 internal_spectral_alpha: float = 0.80,
                 **kwargs):
        super().__init__()
        self.use_decision_fusion = bool(use_decision_fusion)
        self.decision_alpha = float(decision_alpha)
        self.use_internal_spectral = bool(use_internal_spectral)

        if FuncConv1d_cls is not None:
            _func = FuncConv1d_cls
        else:
            mode = str(gabor_mode).strip().lower()
            if mode not in {'1d', '2d'}:
                raise ValueError("DualModal_TFN gabor_mode must be '1d' or '2d'")
            _func = TFconv_Gabor if mode == '1d' else TFconv_Gabor2D
        self.gabor_mode = '2d' if _func is TFconv_Gabor2D else '1d'

        self.vib_branch = _TFN_Backbone(
            in_channels, mid_channel, kernel_size, clamp_flag,
            func_conv_cls=_func,
            gabor2d_kernel_size=gabor2d_kernel_size,
            gabor1d_kernel_size=gabor1d_kernel_size,
        )
        self.aco_branch = _TFN_Backbone(
            in_channels, mid_channel, kernel_size, clamp_flag,
            func_conv_cls=_func,
            gabor2d_kernel_size=gabor2d_kernel_size,
            gabor1d_kernel_size=gabor1d_kernel_size,
        )

        feat_dim = 64
        self.shallow_fusion = FusionGate(feat_dim)
        self.shallow_extractor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.deep_v_extractor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.deep_a_extractor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.deep_fusion = FusionGate(feat_dim)
        self.deep_project = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        fusion_dim = feat_dim * 2
        self.adaptive_weight = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(fusion_dim // 2, out_channels),
        )
        self.vib_head = nn.Linear(feat_dim, out_channels)
        self.aco_head = nn.Linear(feat_dim, out_channels)

        # v5 internal spectral branch: both modalities are used because BJTU's
        # CH20 is weak alone but improves the RBF/MLP spectral boundary when combined
        # with vibration.  The fusion coefficient is learnable and remains in [0,1].
        if self.use_internal_spectral:
            self.spectral_head = _DualSpectralHead(
                out_channels=out_channels,
                bins=internal_spectral_bins,
                hidden=internal_spectral_hidden,
            )
            alpha = min(max(float(internal_spectral_alpha), 1e-4), 1.0 - 1e-4)
            self.spectral_gate_logit = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha)), dtype=torch.float32))
        else:
            self.spectral_head = None
            self.register_parameter('spectral_gate_logit', None)

    def _core_logits(self, x_vib: torch.Tensor, x_aco: torch.Tensor):
        feat_v = self.vib_branch(x_vib)
        feat_a = self.aco_branch(x_aco)

        shallow_fused = self.shallow_fusion(feat_v, feat_a)
        shallow_feat = self.shallow_extractor(shallow_fused)

        deep_v = self.deep_v_extractor(feat_v)
        deep_a = self.deep_a_extractor(feat_a)
        deep_fused = self.deep_fusion(deep_v, deep_a)
        deep_feat = self.deep_project(deep_fused)

        final_feat = torch.cat([shallow_feat, deep_feat], dim=-1)
        weights = self.adaptive_weight(final_feat)
        # 恢复用户原始融合定义，不再使用 v2 的 0.5~1.5 重标定。
        weighted = final_feat * weights
        main_logit = self.classifier(weighted)
        vib_logit = self.vib_head(feat_v)
        aco_logit = self.aco_head(feat_a)
        return main_logit, vib_logit, aco_logit

    def _base_final_logits(self, main_logit, vib_logit, aco_logit):
        base = main_logit
        if self.use_decision_fusion:
            base = main_logit + self.decision_alpha * (vib_logit + aco_logit)
        return base

    def get_spectral_logits(self, x_vib: torch.Tensor, x_aco: torch.Tensor):
        if not self.use_internal_spectral or self.spectral_head is None:
            return None
        return self.spectral_head(x_vib, x_aco)

    def set_spectral_feature_stats(self, mean: torch.Tensor, std: torch.Tensor):
        if self.use_internal_spectral and self.spectral_head is not None:
            self.spectral_head.set_feature_stats(mean, std)

    def spectral_fusion_alpha(self) -> torch.Tensor:
        if self.spectral_gate_logit is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.sigmoid(self.spectral_gate_logit)

    def forward(self, x_vib: torch.Tensor, x_aco: torch.Tensor) -> torch.Tensor:
        main_logit, vib_logit, aco_logit = self._core_logits(x_vib, x_aco)
        base_logit = self._base_final_logits(main_logit, vib_logit, aco_logit)
        if self.use_internal_spectral and self.spectral_head is not None:
            spectral_logit = self.spectral_head(x_vib, x_aco)
            alpha = self.spectral_fusion_alpha()
            # Convex fusion prevents either branch's raw logit scale from exploding.
            return (1.0 - alpha) * base_logit + alpha * spectral_logit
        return base_logit

    def get_branch_logits(self, x_vib: torch.Tensor, x_aco: torch.Tensor):
        """Backward-compatible three-logit API used by older training code."""
        final_logit, vib_logit, aco_logit, _ = self.get_training_logits(x_vib, x_aco)
        return final_logit, vib_logit, aco_logit

    def get_training_logits(self, x_vib: torch.Tensor, x_aco: torch.Tensor):
        """Return final, vibration, acoustic and internal spectral logits."""
        main_logit, vib_logit, aco_logit = self._core_logits(x_vib, x_aco)
        base_logit = self._base_final_logits(main_logit, vib_logit, aco_logit)
        spectral_logit = None
        final_logit = base_logit
        if self.use_internal_spectral and self.spectral_head is not None:
            spectral_logit = self.spectral_head(x_vib, x_aco)
            alpha = self.spectral_fusion_alpha()
            final_logit = (1.0 - alpha) * base_logit + alpha * spectral_logit
        return final_logit, vib_logit, aco_logit, spectral_logit


if __name__ == '__main__':
    # test all the models (单模态)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for Model in [Base_FUNC_CNN, Random_CNN, TFN_Gabor]:
        print('\n\n' + "-" * 50 + '\n' + Model.__name__ + '\n' + "-" * 50)
        models = Model()
        models.to(device)
        # summary 只支持单输入，用 try/except 保护；device 取实际值
        try:
            summary(models, (1, 1024), batch_size=-1, device=str(device))
        except Exception as e:
            print(f"  [summary skipped: {e}]")
        x_test = torch.randn(2, 1, 1024).to(device)
        out_test = models(x_test)
        out_test.mean().backward() #验证反向传播
        print(f"  forward/backward OK, output: {out_test.shape}")

    # test DualModal_TFN模型
    print('\n\n' + "-" * 50 + '\nDualModal_TFN\n' + "-" * 50)
    dual = DualModal_TFN(in_channels=1, out_channels=10, mid_channel=16,
                         use_decision_fusion=True).to(device)
    x_v = torch.randn(4, 1, 1024).to(device)
    x_a = torch.randn(4, 1, 1024).to(device)
    # DualModal 不用 summary，直接跑前向/反向
    out = dual(x_v, x_a)
    out.mean().backward()
    print("DualModal_TFN forward/backward OK, output:", out.shape)
    main_l, vib_l, aco_l = dual.get_branch_logits(x_v, x_a)
    print("get_branch_logits OK:", main_l.shape, vib_l.shape, aco_l.shape)