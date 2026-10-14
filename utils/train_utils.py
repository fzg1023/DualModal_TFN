"""
模型训练全流程：数据集初始化、模型构建、训练/验证/测试、结果可视化
"""

import logging
import copy
import os
import time
import warnings
import torch 
nn = torch.nn
optim = torch.optim
import torch.nn.functional as F
import Models
import Datasets
import matplotlib.pyplot as plt   #绘图
from utils.mysummary import summary
import numpy as np 
import scipy.io as io
import random
import matplotlib as mplb  #绘图配置
import seaborn as sns  #混淆矩阵热力图
import math

try:
    from sklearn.preprocessing import StandardScaler 
    from sklearn.svm import SVC
    import joblib
except Exception:
    StandardScaler = None
    SVC = None
    joblib = None

seed = 999
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class _IndexedDataset(torch.utils.data.Dataset):
    """Return (sample, label, index) without changing the wrapped dataset itself.

    Legacy wrapper retained for compatibility; v5 does not use an external teacher.
    Other models and validation/test loaders keep their original two-item batches.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample, label = self.base_dataset[index]
        return sample, label, index

#训练全流程
class train_utils(object):
    #初始化训练工具类核心参数
    def __init__(self, args, save_dir: str):
        self.args = args #命令行/配置参数
        self.save_dir = save_dir
    #初始化训练核心组件：数据加载器、模型、损失函数、优化器、学习率调度器
    def setup(self):
        args = self.args

        # 考虑gpu or cpu运行状态
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count() #检测可用GPU数量
            logging.info('using %d gpus', self.device_count)
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            # 详细诊断CUDA不可用原因（便于调试）
            logging.warning("GPU is not available. torch.__version__=%s, torch.version.cuda=%s, CUDA_VISIBLE_DEVICES=%s",
                            getattr(torch, '__version__', 'N/A'), getattr(torch.version, 'cuda', 'N/A'),
                            os.environ.get('CUDA_VISIBLE_DEVICES', ''))
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using %d cpu', self.device_count)

        #数据集初始化
        # 兼容连字符命名的数据集（如BJTU-RAO → BJTU_RAO）
        dataset_name = args.data_name.replace('-', '_')
        dataset_cls = getattr(Datasets, dataset_name) #动态加载数据集（解耦数据集逻辑）

        self.datasets = {}
        #提取数据集所需参数（解耦参数传递）
        subargs = {k: getattr(args, k) for k in ['data_dir', 'data_type', 'normlizetype', 'test_size','seed',
'split_seed']}
        # 实例化数据集类，以便读取类别数量等动态属性
        dataset_inst = dataset_cls(subargs)
        self.dataset_inst = dataset_inst
    
        # 区分训练/验证 SNR。DualModal_TFN 的强噪声模式可在归一化之后、GPU 上
        # 按每个样本/每个模态独立功率加噪，避免 UORED 两通道量纲差异造成错误 SNR。
        snr_train = getattr(self.args, 'train_SNR', getattr(self.args, 'SNR', None))
        snr_val = getattr(self.args, 'val_SNR', snr_train)
        is_dual = (args.model_name == 'DualModal_TFN')
        self.is_dual = is_dual
        self.dual_robust_snr = bool(
            is_dual
            and snr_train is not None
            and snr_val is not None
            and abs(float(snr_train) - float(snr_val)) < 1e-12
        )
        self.dual_balanced_snr = bool(
            self.dual_robust_snr and getattr(args, 'dual_balanced_snr', True)
        )
        self._requested_train_snr = snr_train
        self._requested_val_snr = snr_val
        # v5 removes the external SVM teacher/KD path.  The spectral representation
        # is now a learnable submodule inside DualModal_TFN itself.
        self.dual_teacher_requested = False
        self.use_internal_spectral = bool(
            self.dual_robust_snr and getattr(args, 'dual_internal_spectral', True)
        )

        dataset_snr_train = None if self.dual_balanced_snr else snr_train
        dataset_snr_val = None if self.dual_balanced_snr else snr_val
        # 准备数据（返回训练/验证/测试集+标签名）
        (self.datasets['train'], self.datasets['val'], self.datasets['test']), self.label_name = dataset_inst.data_preprare(
            signal_size=args.data_signalsize,
            SNR_train=dataset_snr_train,
            SNR_val=dataset_snr_val)

        # 为非空数据集创建DataLoader（空数据集设为None）
        from torch.utils.data.sampler import SubsetRandomSampler 
        self.dataloaders = {}
        for x in ['train', 'val', 'test']:
            ds = self.datasets.get(x, None)
            if ds is None or len(ds) == 0:
                logging.warning('Dataset %s is empty or missing; skipping dataloader creation.', x)
                self.dataloaders[x] = None
                continue
            if x == 'train':
                loader_ds = _IndexedDataset(ds) if self.dual_teacher_requested else ds
                self.dataloaders[x] = torch.utils.data.DataLoader(loader_ds, batch_size=args.batch_size,
                                                                 shuffle=True, #训练集打乱
                                                                 num_workers=args.num_workers,
                                                                 pin_memory=(self.device.type == 'cuda'))# GPU下开启pin_memory加速

            else:
                # 验证/测试集固定顺序
                self.dataloaders[x] = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                                 shuffle=False,
                                                                 num_workers=args.num_workers,
                                                                 pin_memory=(self.device.type == 'cuda'))
        logging.info(f"dataset_train:{len(self.datasets['train']):d}, dataset_val:{len(self.datasets['val']):d}, dataset_test:{len(self.datasets['test']):d}")
        # 记录 SNR 增强信息
        if snr_train is not None:
            if self.dual_balanced_snr:
                logging.info(
                    'DualModal balanced SNR active: %s dB, applied after dataset normalization '
                    'with independent per-modality power on device', snr_train)
            else:
                logging.info('SNR augmentation active for training: %s dB (dataset transform)', snr_train)
        else:
            logging.info('No SNR augmentation for training')

        # 模型初始化
        if self.dual_robust_snr:
            logging.info(
                "DualModal low-SNR robust mode enabled: train_SNR=val_SNR=%s dB",
                snr_train,
            )
        if is_dual:
            dual_gabor_mode = str(getattr(args, 'dual_gabor_mode', 'auto')).strip().lower()
            if dual_gabor_mode == 'auto':
                # Fast noisy path: 1D Gabor + a tiny FFT residual classifier.
                # Users can still force the original slower 2D STFT-Gabor path via --dual_gabor_mode 2d.
                dual_gabor_mode = '1d' if self.dual_robust_snr else '2d'
            logging.info('DualModal Gabor front-end selected: %s', dual_gabor_mode)
            self.model = getattr(Models, args.model_name)(
                in_channels=1,
                out_channels=dataset_inst.num_classes,
                kernel_size=args.kernel_size,
                clamp_flag=args.clamp_flag,
                mid_channel=args.mid_channel,
                use_decision_fusion=getattr(args, 'use_decision_fusion', True),
                decision_alpha=getattr(args, 'decision_alpha', 0.1),
                gabor2d_kernel_size=getattr(args, 'gabor2d_kernel_size', 15),
                gabor1d_kernel_size=getattr(args, 'gabor1d_kernel_size', 63),
                gabor_mode=dual_gabor_mode,
                use_internal_spectral=self.use_internal_spectral,
                internal_spectral_bins=getattr(args, 'internal_spectral_bins', 128),
                internal_spectral_hidden=getattr(args, 'internal_spectral_hidden', 128),
                internal_spectral_alpha=getattr(args, 'internal_spectral_alpha', 0.80),
            )
        else:
            # 对于单模态模型，模型实际接收单通道输入（训练时会对双通道数据取第0通道）。
            # 因此这里强制设置 in_channels=1，避免模型期望与实际输入通道不匹配导致的运行时错误。
            model_kwargs = dict(
                in_channels=1,
                out_channels=dataset_inst.num_classes,
                kernel_size=args.kernel_size,
                clamp_flag=args.clamp_flag,
                mid_channel=args.mid_channel)
            # TFN_Gabor 专用参数
            if args.model_name == 'TFN_Gabor':
               model_kwargs['gabor1d_kernel_size'] = \
                args.gabor1d_kernel_size
            self.model = getattr(Models,args.model_name)(**model_kwargs)

            if args.model_name == 'TFN_Gabor':
               self.model.funconv.superparams.requires_grad_(False)
               logging.info("Stage-1: freeze Gabor parameters for %d epochs", args.gabor_warmup_epochs)
    
        self.model.to(self.device) #模型移至指定设备（GPU/CPU）
        logging.info("[MODEL CHECK] model_name = %s",
                     args.model_name)
        logging.info("[MODEL CHECK] model_class = %s",
                     self.model.__class__.__name__)
        logging.info("[MODEL CHECK] trainable_params = %d",
                     sum(p.numel()
                         for p in self.model.parameters()
                         if p.requires_grad))

        # 总结模型并记录模型信息
        try:
            # 对于双模态模型，跳过摘要分支（该分支需要两张输入张量）；
            # 对于单模态模型，确保输入尺寸为单通道，避免模型参数汇总（summary）时出现通道维度不匹配问题
            if self.is_dual:
                logging.info('Skipping model summary for dual-modal model %s', args.model_name)
            else:
                sample_shape = self.datasets['train'][0][0].shape
                #样本形状通常为（通道数，序列长度）。若数据集提供双通道样本（通道数等于 2）
                # 仅使用单个通道生成汇总数据，以适配单模态模型的预期输入。
                if isinstance(sample_shape, tuple) and len(sample_shape) >= 2:
                    ch = sample_shape[0]
                    if ch == 2:
                        input_size = (1, sample_shape[1])
                    else:
                        input_size = sample_shape
                else:
                    input_size = sample_shape

                info = summary(self.model, input_size, batch_size=-1, device=self.device.type)
                for item in info.split('\n'):
                    logging.info(item)
        except Exception as e:
            logging.exception("Model summary failed on device %s: %s", self.device, e)
            # 尝试回退到 CPU 上运行 summary 以便至少获取模型结构信息（仅针对单模态）
            if not self.is_dual:
                try:
                    logging.info("Trying summary on CPU as fallback...")
                    self.model.to(torch.device('cpu'))
                    info = summary(self.model, input_size, batch_size=-1, device='cpu')
                    for item in info.split('\n'):
                        logging.info(item)
                except Exception:
                    logging.exception("Model summary also failed on CPU.")
                finally:
                    # 恢复模型到原始设备
                    try:
                        self.model.to(self.device)
                    except Exception:
                        logging.exception("Failed to move model back to device %s", self.device)
            else:
                logging.info('Summary skipped for dual model after failure: %s', e)
        #损失函数（分类任务用交叉熵）
        self.criterion = nn.CrossEntropyLoss()

        # v5: one-time train-split-only supervised warm start for the INTERNAL
        # spectral head.  No validation/test labels or external teacher are used.
        if self.is_dual and self.use_internal_spectral:
            self._pretrain_internal_spectral_head(dataset_inst)

        # 优化器（支持SGD/Adam/RMSprop
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            if args.model_name == 'TFN_Gabor':
               # Gabor 超参数单独取出
               gabor_params = [self.model.funconv.superparams]
               gabor_param_ids = {id(p) for p in gabor_params}
               # CNN 其余参数
               backbone_params = [p for p in self.model.parameters()
                                  if id(p) not in gabor_param_ids]
               self.optimizer = optim.Adam(backbone_params, lr=args.lr, weight_decay=args.weight_decay)
               self.gabor_optimizer = optim.SGD(gabor_params, lr=args.gabor_lr, momentum=0.0, weight_decay=0.0)
               logging.info('TFN_Gabor optimizer: CNN lr=%g, Gabor lr=%g', args.lr, args.gabor_lr)
            elif args.model_name == 'DualModal_TFN':
               gabor_params = [
                   self.model.vib_branch.funconv.superparams,
                   self.model.aco_branch.funconv.superparams,
               ]
               gabor_param_ids = {id(p) for p in gabor_params}
               backbone_params = [p for p in self.model.parameters() if id(p) not in gabor_param_ids]
               # v3 fix: 1D Gabor can use the conservative 1e-5 rate, but 2D Gabor
               # has randomly initialized theta and a learnable sigma; 1e-5 effectively freezes it.
               if getattr(self.model, 'gabor_mode', '1d') == '2d':
                   dual_gabor_lr = float(getattr(args, 'dual_gabor2d_lr', 5e-4))
               else:
                   dual_gabor_lr = float(args.gabor_lr)
               self._dual_gabor_lr = dual_gabor_lr
               # Keep references for targeted DualModal Gabor gradient clipping.
               self._dual_gabor_params = gabor_params
               self.optimizer = optim.Adam([
                   {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                   {'params': gabor_params, 'lr': dual_gabor_lr, 'weight_decay': 0.0},
               ])
               logging.info(
                   'DualModal_TFN optimizer groups: backbone lr=%g, %s-Gabor lr=%g',
                   args.lr, getattr(self.model, 'gabor_mode', 'unknown'), dual_gabor_lr
               )
               logging.info(
                   'DualModal stability: global grad clip=%g, Gabor grad clip=%g',
                   float(getattr(args, 'dual_grad_clip', 5.0)),
                   float(getattr(args, 'dual_gabor_grad_clip', 1.0))
               )
            else:
                 self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                           weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # 学习率调度器（支持step/exp/stepLR/fix）
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # 混淆矩阵初始化（训练/验证/测试各一个）
        self.c_matrix = {phase: np.zeros([dataset_inst.num_classes, dataset_inst.num_classes]) for phase in ['train', 'val', 'test']}

        # v5: the external sklearn/SVM teacher is disabled.  Keep legacy attributes
        # as None/False so old helper functions remain harmless if imported.
        self.spectral_scaler = None
        self.spectral_svc = None
        self.teacher_train_probs = None
        self.dual_spectral_teacher = False

        # EMA is also DualModal/noisy-only; other models retain their exact behavior.
        self.use_dual_ema = bool(
            self.dual_robust_snr and getattr(args, 'dual_ema', True)
        )
        self.ema_model = None
        self.ema_updates = 0
        if self.use_dual_ema:
            self._init_dual_ema()
    
    def _pretrain_internal_spectral_head(self, dataset_inst):
        """Warm-start v5's internal spectral MLP using TRAIN split labels only.

        This is part of DualModal_TFN training, not an external teacher.  Features are
        generated from several independent target-SNR views of the raw TRAIN windows;
        their mean/std are stored as buffers in the model and are therefore available
        to normal PyTorch inference without sklearn.
        """
        if not getattr(self.model, 'use_internal_spectral', False) or getattr(self.model, 'spectral_head', None) is None:
            return
        raw_splits = getattr(dataset_inst, 'raw_splits', None)
        train_pd = raw_splits.get('train') if isinstance(raw_splits, dict) else None
        if train_pd is None or len(train_pd) == 0:
            logging.warning('v5 internal spectral pretrain skipped: dataset exposes no raw TRAIN split.')
            return
        try:
            x = np.stack(train_pd['data'].to_numpy()).astype(np.float32)
            y = train_pd['label'].to_numpy(dtype=np.int64)
        except Exception as e:
            logging.exception('v5 failed to build internal spectral pretrain data: %s', e)
            return
        if x.ndim != 3 or x.shape[1] < 2 or len(x) != len(self.datasets['train']):
            logging.warning('v5 internal spectral pretrain skipped: unexpected raw shape/length %s.', x.shape)
            return

        snr_db = float(self._requested_train_snr)
        repeats = max(1, int(getattr(self.args, 'internal_spectral_pretrain_repeats', 6)))
        pre_epochs = max(0, int(getattr(self.args, 'internal_spectral_pretrain_epochs', 15)))
        bins = int(getattr(self.args, 'internal_spectral_bins', 128))
        pre_lr = float(getattr(self.args, 'internal_spectral_pretrain_lr', 1e-3))
        if pre_epochs <= 0:
            logging.info('v5 internal spectral pretrain disabled (epochs=0).')
            return

        t0 = time.time()
        # Match runtime robust-SNR policy: per window and per modality RMS normalization,
        # followed by independent AWGN at the requested SNR.
        x = x - x.mean(axis=-1, keepdims=True)
        x = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)).clip(1e-6, None)
        signal_power = np.mean(x * x, axis=-1, keepdims=True).clip(1e-8, None)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        base_seed = int(getattr(self.args, 'seed', 999)) + 520000

        feats, labels = [], []
        for rep in range(repeats):
            rng = np.random.default_rng(base_seed + rep)
            noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(noise_power).astype(np.float32)
            noisy = x + noise
            fv = self._spectral_features_np(noisy[:, 0:1, :], bins=bins)
            fa = self._spectral_features_np(noisy[:, 1:2, :], bins=bins)
            feats.append(np.concatenate([fv, fa], axis=1).astype(np.float32))
            labels.append(y)
        feat_np = np.concatenate(feats, axis=0).astype(np.float32)
        label_np = np.concatenate(labels, axis=0).astype(np.int64)

        feat_mean = feat_np.mean(axis=0).astype(np.float32)
        feat_std = feat_np.std(axis=0).astype(np.float32)
        feat_std = np.clip(feat_std, 1e-5, None)
        self.model.set_spectral_feature_stats(
            torch.from_numpy(feat_mean).to(self.device),
            torch.from_numpy(feat_std).to(self.device),
        )

        feat_cpu = torch.from_numpy(feat_np)
        label_cpu = torch.from_numpy(label_np)
        head_params = list(self.model.spectral_head.net.parameters())
        pre_opt = optim.AdamW(head_params, lr=pre_lr, weight_decay=1e-4)
        batch_size = min(max(int(getattr(self.args, 'batch_size', 32)) * 4, 64), 256)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(base_seed + 999)

        self.model.spectral_head.train()
        last_loss = 0.0
        for ep in range(pre_epochs):
            perm = torch.randperm(len(feat_cpu), generator=gen)
            loss_sum = 0.0
            correct = 0
            seen = 0
            for st in range(0, len(feat_cpu), batch_size):
                idx = perm[st:st + batch_size]
                fb = feat_cpu.index_select(0, idx).to(self.device, non_blocking=True)
                yb = label_cpu.index_select(0, idx).to(self.device, non_blocking=True)
                pred = self.model.spectral_head.classify_features(fb)
                loss = self.criterion(pred, yb)
                pre_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
                pre_opt.step()
                loss_sum += float(loss.detach().item()) * len(idx)
                correct += int((pred.argmax(dim=1) == yb).sum().item())
                seen += len(idx)
            last_loss = loss_sum / max(1, seen)
            if ep in {0, pre_epochs - 1}:
                logging.info(
                    'v5 internal spectral pretrain epoch %d/%d: loss=%.4f acc=%.2f%%',
                    ep + 1, pre_epochs, last_loss, 100.0 * correct / max(1, seen)
                )
        self.model.spectral_head.eval()
        logging.info(
            'v5 internal spectral branch ready: TRAIN ONLY, samples=%d x repeats=%d, bins=%d, '
            'feature_dim=%d, setup_time=%.2fs; NO external teacher/KD.',
            len(y), repeats, bins, feat_np.shape[1], time.time() - t0
        )

    @staticmethod
    def _spectral_features_np(x_vib, bins=256):
        """Robust vibration features: pooled log spectrum + cepstrum + autocorrelation.

        x_vib: numpy array (N,1,L) or (N,L).  Each sample is standardized again so
        the expert uses spectral shape rather than absolute sensor scale.
        """
        x = np.asarray(x_vib, dtype=np.float32)
        if x.ndim == 3:
            x = x[:, 0, :]
        x = x - x.mean(axis=-1, keepdims=True)
        std = np.sqrt(np.mean(x * x, axis=-1, keepdims=True)).clip(1e-6, None)
        z = x / std

        mag = np.abs(np.fft.rfft(z, axis=-1)).astype(np.float32)[:, 1:]
        nfreq = mag.shape[1]
        bins = int(max(8, min(int(bins), nfreq)))
        if nfreq % bins == 0:
            pooled = mag.reshape(mag.shape[0], bins, nfreq // bins).mean(axis=-1)
        else:
            edges = np.linspace(0, nfreq, bins + 1).astype(np.int64)
            pooled = np.stack([
                mag[:, edges[i]:max(edges[i] + 1, edges[i + 1])].mean(axis=1)
                for i in range(bins)
            ], axis=1)
        log_spec = np.log1p(pooled)

        log_mag = np.log1p(mag)
        cep = np.fft.rfft(log_mag, axis=-1).real[:, :24].astype(np.float32)
        ac = np.fft.irfft(mag * mag, axis=-1).real[:, :24].astype(np.float32)
        ac = ac / np.clip(ac[:, :1], 1e-6, None)
        return np.concatenate([log_spec, cep, ac], axis=1).astype(np.float32)

    @staticmethod
    def _spectral_features_torch(x_vib, bins=256):
        """Torch equivalent of _spectral_features_np for online expert inference."""
        x = x_vib
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x0 = x - x.mean(dim=-1, keepdim=True)
        std = x0.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        z = x0 / std
        mag = torch.fft.rfft(z, dim=-1).abs()[..., 1:]
        nfreq = mag.size(-1)
        bins = int(max(8, min(int(bins), int(nfreq))))
        pooled = F.adaptive_avg_pool1d(mag, bins)
        log_spec = torch.log1p(pooled).squeeze(1)
        log_mag = torch.log1p(mag.squeeze(1))
        cep = torch.fft.rfft(log_mag, dim=-1).real[:, :32]
        ac = torch.fft.irfft(mag.squeeze(1).pow(2), dim=-1).real[:, :32]
        ac = ac / ac[:, :1].clamp_min(1e-6)
        return torch.cat([log_spec, cep, ac], dim=-1)

    @staticmethod
    def _expand_svc_scores(score, classes, num_classes):
        """Map sklearn decision_function columns to [0, num_classes)."""
        score = np.asarray(score, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.int64)
        if score.ndim == 1:
            score = np.stack([-score, score], axis=1)
        out = np.full((score.shape[0], int(num_classes)), -8.0, dtype=np.float32)
        for col, cls in enumerate(classes.tolist()):
            if 0 <= int(cls) < out.shape[1] and col < score.shape[1]:
                out[:, int(cls)] = score[:, col]
        return out

    def _fit_dual_spectral_teacher(self, dataset_inst):
        """Fit a train-only spectral SVM and cache soft logits for each train sample.

        Data policy:
          * fitting uses only raw_splits['train'] and its labels;
          * target logits are precomputed only for those same train indices;
          * validation/test samples and labels are never passed to the teacher;
          * validation/test inference is pure DualModal_TFN (EMA weights when enabled).
        """
        if StandardScaler is None or SVC is None:
            logging.warning('scikit-learn unavailable; disabling v4 spectral teacher.')
            self.dual_spectral_teacher = False
            return
        raw_splits = getattr(dataset_inst, 'raw_splits', None)
        if not raw_splits or 'train' not in raw_splits:
            logging.warning('Dataset does not expose raw_splits; disabling v4 spectral teacher.')
            self.dual_spectral_teacher = False
            return
        train_pd = raw_splits['train']
        if train_pd is None or len(train_pd) == 0:
            self.dual_spectral_teacher = False
            return
        try:
            x = np.stack(train_pd['data'].to_list()).astype(np.float32)
            y = train_pd['label'].to_numpy(dtype=np.int64)
        except Exception as e:
            logging.exception('Failed to build v4 teacher data: %s', e)
            self.dual_spectral_teacher = False
            return
        if x.ndim != 3 or x.shape[1] < 1:
            logging.warning('Unexpected raw train shape for v4 teacher: %s', x.shape)
            self.dual_spectral_teacher = False
            return
        if len(x) != len(self.datasets['train']):
            logging.warning(
                'Teacher/raw train length mismatch (%d vs %d); disabling teacher to avoid index leakage.',
                len(x), len(self.datasets['train']))
            self.dual_spectral_teacher = False
            return

        # Per-window/per-modality normalization, matching the robust SNR policy.
        x = x - x.mean(axis=-1, keepdims=True)
        x = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)).clip(1e-6, None)
        snr_db = float(self._requested_train_snr)
        fit_repeats = max(1, int(getattr(self.args, 'teacher_fit_repeats', 8)))
        target_repeats = max(1, int(getattr(self.args, 'teacher_target_repeats', 4)))
        bins = max(32, int(getattr(self.args, 'teacher_bins', 256)))
        c_value = float(getattr(self.args, 'teacher_C', 0.5))
        base_seed = int(getattr(self.args, 'seed', 999)) + 410000
        signal_power = np.mean((x - x.mean(axis=-1, keepdims=True)) ** 2,
                               axis=-1, keepdims=True).clip(1e-8, None)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))

        feature_parts, label_parts = [], []
        for rep in range(fit_repeats):
            rng = np.random.default_rng(base_seed + rep)
            noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(noise_power).astype(np.float32)
            noisy = x + noise
            feature_parts.append(self._spectral_features_np(noisy[:, 0:1, :], bins=bins))
            label_parts.append(y)
        feats = np.concatenate(feature_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)

        self.spectral_scaler = StandardScaler()
        feats_scaled = self.spectral_scaler.fit_transform(feats)
        self.spectral_svc = SVC(
            C=c_value, kernel='rbf', gamma='scale', decision_function_shape='ovr',
            probability=True, cache_size=512, random_state=int(getattr(self.args, 'seed', 999)),
        )
        t0 = time.time()
        self.spectral_svc.fit(feats_scaled, labels)
        fit_acc = float(np.mean(self.spectral_svc.predict(feats_scaled) == labels))

        # Average calibrated class probabilities over several independent target-SNR views.
        # Unlike raw SVC OVR margins, predict_proba has a probability interpretation suitable for KL KD.
        prob_sum = np.zeros((len(x), int(dataset_inst.num_classes)), dtype=np.float32)
        for rep in range(target_repeats):
            rng = np.random.default_rng(base_seed + 10000 + rep)
            noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(noise_power).astype(np.float32)
            noisy = x + noise
            f = self._spectral_features_np(noisy[:, 0:1, :], bins=bins)
            f = self.spectral_scaler.transform(f)
            raw_prob = self.spectral_svc.predict_proba(f).astype(np.float32)
            expanded = np.zeros((len(x), int(dataset_inst.num_classes)), dtype=np.float32)
            for j, cls in enumerate(self.spectral_svc.classes_):
                expanded[:, int(cls)] = raw_prob[:, j]
            prob_sum += expanded
        prob = prob_sum / float(target_repeats)
        prob = prob / prob.sum(axis=1, keepdims=True).clip(1e-8, None)
        self.teacher_train_probs = torch.from_numpy(prob.astype(np.float32))  # CPU cache, TRAIN ONLY

        teacher_pred = prob.argmax(axis=1)
        target_acc = float(np.mean(teacher_pred == y))
        mean_conf = float(prob.max(axis=1).mean())
        high_conf_correct = float(np.mean((teacher_pred == y) & (prob.max(axis=1) >= float(getattr(self.args, 'distill_conf_floor', 0.70)))))
        logging.info(
            'v4.1 TRAIN-ONLY calibrated spectral teacher ready: train=%d, fit_repeats=%d, target_repeats=%d, '
            'bins=%d, support_vectors=%d, fit_acc=%.4f, cached_target_acc=%.4f, '
            'mean_conf=%.4f, high_conf_correct=%.4f, setup_time=%.2fs',
            len(y), fit_repeats, target_repeats, bins, len(self.spectral_svc.support_),
            fit_acc, target_acc, mean_conf, high_conf_correct, time.time() - t0)
        logging.info('v4.1 policy: calibrated teacher is TRAIN-ONLY; validation/test are pure DualModal_TFN.')
        if joblib is not None:
            try:
                joblib.dump(
                    {'scaler': self.spectral_scaler, 'svc': self.spectral_svc,
                     'snr_db': snr_db, 'bins': bins, 'train_only': True},
                    os.path.join(self.save_dir, 'v4_train_only_spectral_teacher.joblib'))
            except Exception:
                logging.exception('Failed to save v4 train-only teacher artifact')

    def _teacher_probs_for_indices(self, sample_indices, device, dtype):
        if (not self.dual_spectral_teacher or self.teacher_train_probs is None
                or sample_indices is None):
            return None
        if torch.is_tensor(sample_indices):
            idx = sample_indices.detach().cpu().long()
        else:
            idx = torch.as_tensor(sample_indices, dtype=torch.long)
        return self.teacher_train_probs.index_select(0, idx).to(device=device, dtype=dtype)

    def _soft_distill_loss(self, student_logits, teacher_prob, labels):
        """v4.1: calibrated, label-aware and confidence-gated train-only KD."""
        if teacher_prob is None:
            return student_logits.new_tensor(0.0)
        temperature = max(0.5, float(getattr(self.args, 'distill_temperature', 1.5)))
        conf_floor = float(getattr(self.args, 'distill_conf_floor', 0.70))
        conf_floor = min(max(conf_floor, 0.0), 0.99)
        with torch.no_grad():
            p = teacher_prob.clamp_min(1e-7)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-7)
            teacher_conf, teacher_pred = p.max(dim=-1)
            # Never distill a teacher mistake on a labelled TRAIN sample.
            correct_mask = teacher_pred.eq(labels).float()
            conf_w = ((teacher_conf - conf_floor) / max(1e-6, 1.0 - conf_floor)).clamp(0.0, 1.0)
            conf_w = conf_w * correct_mask
            if float(conf_w.sum().item()) <= 0.0:
                return student_logits.new_tensor(0.0)
            # Mild label anchoring keeps the true class dominant even if calibration is imperfect.
            one_hot = F.one_hot(labels, num_classes=student_logits.shape[-1]).to(dtype=p.dtype)
            p = 0.85 * p + 0.15 * one_hot
            # Temperature-soften a calibrated probability distribution.
            teacher_t = p.pow(1.0 / temperature)
            teacher_t = teacher_t / teacher_t.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        per_sample = F.kl_div(student_log_prob, teacher_t, reduction='none').sum(dim=-1)
        return (per_sample * conf_w).sum() / conf_w.sum().clamp_min(1e-6) * (temperature ** 2)

    def _init_dual_ema(self):
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_updates = 0
        self.ema_started = False
        logging.info('v5 DualModal EMA enabled: start_epoch=%d, max_decay=%.6f', int(getattr(self.args, 'ema_start_epoch', 5)), float(getattr(self.args, 'ema_decay', 0.990)))

    @torch.no_grad()
    def _update_dual_ema(self, epoch):
        if not self.use_dual_ema or self.ema_model is None:
            return
        start_epoch = max(0, int(getattr(self.args, 'ema_start_epoch', 5)))
        if epoch < start_epoch:
            return
        # At EMA start copy the already-trained live model exactly; do not average random init.
        if not getattr(self, 'ema_started', False):
            self.ema_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self.ema_started = True
            self.ema_updates = 0
            return
        self.ema_updates += 1
        max_decay = min(max(float(getattr(self.args, 'ema_decay', 0.990)), 0.0), 0.99999)
        warm_decay = (1.0 + self.ema_updates) / (10.0 + self.ema_updates)
        decay = min(max_decay, warm_decay)
        model_params = dict(self.model.named_parameters())
        for name, ema_p in self.ema_model.named_parameters():
            src = model_params[name].detach()
            ema_p.mul_(decay).add_(src, alpha=1.0 - decay)
        # BN running statistics/counters should reflect the current model directly rather
        # than being double-smoothed by EMA.
        model_buffers = dict(self.model.named_buffers())
        for name, ema_b in self.ema_model.named_buffers():
            src = model_buffers[name].detach()
            ema_b.copy_(src)

    def _split_inputs(self, inputs):
        """严格拆分振动/声学输入，避免静默复制通道造成伪双模态实验。"""
        if self.is_dual:
            if inputs.ndim != 3 or inputs.shape[1] < 2:
                raise ValueError(
                    "DualModal_TFN requires input shape (B,2,L) with channel0=vibration and channel1=acoustic; "
                    f"got {tuple(inputs.shape)}"
                )
            return inputs[:, 0:1, :], inputs[:, 1:2, :]

        # 单模态：由 --single_modality 明确选择通道
        modality = getattr(self.args, 'single_modality', 'vib')
        if inputs.ndim == 3 and inputs.shape[1] >= 2:
            idx = 0 if modality == 'vib' else 1
            return inputs[:, idx:idx+1, :]
        if modality == 'aco' and (inputs.ndim != 3 or inputs.shape[1] < 2):
            raise ValueError("single_modality=aco requires a two-channel dataset containing acoustic channel 1.")
        return inputs

    def _apply_dual_balanced_awgn(self, inputs, snr_db, phase, epoch, batch_idx):
        """Apply exact AWGN independently to every sample and modality.

        This runs after the dataset transform (normally mean-std), so UORED's very
        different accelerometer/acoustic raw scales cannot distort the requested SNR.
        Validation/test noise is deterministic across epochs for stable model selection.
        """
        if not self.dual_balanced_snr or snr_db is None:
            return inputs
        if inputs.ndim != 3 or inputs.shape[1] < 2:
            return inputs

        # Centered signal power for each (sample, channel), shape (B,C,1).
        centered = inputs - inputs.mean(dim=-1, keepdim=True)
        signal_power = centered.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
        noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))

        generator = None
        if phase != 'train':
            # Keep validation/test corruption fixed across epochs.  A changing validation
            # noise realization can move accuracy by several points on BJTU's small val set.
            phase_offset = 100000 if phase == 'val' else 200000
            try:
                generator = torch.Generator(device=inputs.device)
            except TypeError:
                generator = torch.Generator(device=inputs.device.type)
            generator.manual_seed(int(getattr(self.args, 'seed', 999)) + phase_offset + int(batch_idx))

        noise = torch.randn(
            inputs.shape, device=inputs.device, dtype=inputs.dtype, generator=generator
        ) * noise_power.sqrt()
        return inputs + noise

    #执行模型的训练、验证、测试流程：按 epoch 迭代训练，记录损失 / 精度，保存最优模型，最终在测试集评估，输出训练日志。
    def train(self):
        """
        训练+验证+测试全流程
        """
        args = self.args

        best_acc = 0.0 #最优验证集精度（用于保存最优模型）
        step_start = time.time() #批次计时起点
        # 初始化最优模型保存变量（避免max_epoch=1时未定义）
        save_best_data = None
        save_best_data_dir = None
        # 将实例展示出来，以便后续plot_save能够访问到最优状态
        self.save_best_data = None
        self.save_best_data_dir = None
        #训练记录（epoch级：训练/验证/测试损失/精度 + 最优epoch）
        self.Records = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [],
                "best_epoch": 0}  # epoch-wise records (include test)
        #批次级记录（训练损失/精度）
        self.MinorRecords = {"train_loss": [], "train_acc": []}  # batch-wise records

        # 通过轮次训练模型
        for epoch in range(args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # =====================================================
# Stage 2：达到warmup epoch后，解冻Gabor进行小步微调
# =====================================================
            if (args.model_name == 'TFN_Gabor'and epoch == args.gabor_warmup_epochs):

                 self.model.funconv.superparams.requires_grad_(True)

                 logging.info("Stage-2: unfreeze Gabor parameters ""at epoch %d, Gabor lr = %.2e", epoch,args.gabor_lr )

                 
            # 记录学习率
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))
            #遍历训练/验证阶段
            for phase in ['train', 'val']:

                # 跳过空数据集
                if phase not in self.datasets or self.datasets[phase] is None or len(self.datasets[phase]) == 0:
                    logging.warning('Skipping phase %s because dataset is empty.', phase)
                    continue
                # 初始化批次/epoch级变量
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                batch_acc = 0
                batch_loss = 0.0
                batch_count = 0
                epoch_neural_correct = 0.0
                epoch_live_correct = 0.0
                epoch_spectral_correct = 0.0

                # v5: training updates the live model; validation uses EMA weights when
                # enabled.  Both paths are the same v5 DualModal_TFN architecture.
                if phase == 'train':
                    self.model.train()
                    forward_model = self.model
                else:
                    self.model.eval()
                    forward_model = self.ema_model if (self.use_dual_ema and self.ema_model is not None and getattr(self, 'ema_started', False)) else self.model
                    forward_model.eval()

                epoch_kd_sum = 0.0
                epoch_kd_count = 0
                # 按批次训练或测试
                for batch_idx, batch_data in enumerate(self.dataloaders[phase]):
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                        inputs, labels, sample_indices = batch_data
                    else:
                        inputs, labels = batch_data
                        sample_indices = None
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    if self.is_dual and self.dual_balanced_snr:
                        phase_snr = self._requested_train_snr if phase == 'train' else self._requested_val_snr
                        inputs = self._apply_dual_balanced_awgn(inputs, phase_snr, phase, epoch, batch_idx)
                    # 梯度控制：训练阶段计算梯度，验证阶段禁用（节省显存）
                    with torch.set_grad_enabled(phase == 'train'):
                        # 前向传播：区分 双模态 or 单模态
                        if self.is_dual:
                            x_vib, x_aco = self._split_inputs(inputs)
                            neural_logit = None
                            if phase == 'train':
                                spectral_logit = None
                                if hasattr(self.model, 'get_training_logits'):
                                    neural_logit, vib_logit, aco_logit, spectral_logit = self.model.get_training_logits(x_vib, x_aco)
                                elif getattr(self.args, 'aux_loss', False):
                                    neural_logit, vib_logit, aco_logit = self.model.get_branch_logits(x_vib, x_aco)
                                else:
                                    neural_logit = self.model(x_vib, x_aco)
                                # v5 final classifier is one DualModal_TFN model, including its internal spectral branch.
                                logits = neural_logit
                                loss = self.criterion(neural_logit, labels)

                                if getattr(self.args, 'aux_loss', False):
                                    aux_w = float(getattr(self.args, 'aux_weight', 0.2))
                                    if self.dual_robust_snr:
                                        vib_aux_w = aux_w * 0.50
                                        aco_aux_w = aux_w * 0.10
                                    else:
                                        vib_aux_w = aux_w
                                        aco_aux_w = aux_w
                                    loss = (loss
                                            + vib_aux_w * self.criterion(vib_logit, labels)
                                            + aco_aux_w * self.criterion(aco_logit, labels))

                                if spectral_logit is not None:
                                    spectral_aux_w = max(0.0, float(getattr(self.args, 'internal_spectral_aux_weight', 0.60)))
                                    if spectral_aux_w > 0.0:
                                        loss = loss + spectral_aux_w * self.criterion(spectral_logit, labels)
                                    epoch_spectral_correct += torch.eq(
                                        spectral_logit.argmax(dim=1), labels
                                    ).float().sum().item()
                            else:
                                # Validation is one PURE DualModal_TFN model (including its internal
                                # spectral submodule). Evaluate EMA and live weights on the same fixed noise.
                                spectral_logit = None
                                if hasattr(forward_model, 'get_training_logits'):
                                    neural_logit, _, _, spectral_logit = forward_model.get_training_logits(x_vib, x_aco)
                                else:
                                    neural_logit = forward_model(x_vib, x_aco)
                                logits = neural_logit
                                loss = self.criterion(logits, labels)
                                if spectral_logit is not None:
                                    epoch_spectral_correct += torch.eq(
                                        spectral_logit.argmax(dim=1), labels
                                    ).float().sum().item()
                                if self.use_dual_ema and self.ema_model is not None and getattr(self, 'ema_started', False):
                                    live_logit = self.model(x_vib, x_aco)
                                    epoch_live_correct += torch.eq(live_logit.argmax(dim=1), labels).float().sum().item()
                                else:
                                    epoch_live_correct += torch.eq(logits.argmax(dim=1), labels).float().sum().item()
                        else:
                            inputs_single = self._split_inputs(inputs)
                            logits = self.model(inputs_single)
                            loss = self.criterion(logits, labels)
                            # =====================================================
                            # Gabor超参数保持正则
                            # 仅在Gabor解冻以后启用
                            # =====================================================
                            if (args.model_name == 'TFN_Gabor'
                                and phase == 'train'
                                and epoch >= args.gabor_warmup_epochs
                                and args.gabor_reg > 0 
                                and hasattr(self.model.funconv,'superparams_init')):

    # 当前中心频率
                                current_f0 = (self.model.funconv 
                                              .superparams[..., 0] )

    # 初始中心频率
                                initial_f0 = (self.model.funconv
                                              .superparams_init[..., 0])

    # 防止f0偏离有效初始化太远
                                gabor_reg_loss = ( current_f0 - initial_f0 ).pow(2).mean()

                                loss = (loss
                                        + args.gabor_reg
                                         * gabor_reg_loss)
                        # DualModal 低 SNR 模式：轻量约束两路 Gabor 参数不要偏离有效初始化。
                        # 仅在目标模式启用，不影响其他模型/clean 实验。
                        if (self.is_dual and self.dual_robust_snr and phase == 'train'
                                and getattr(self.model, 'gabor_mode', '1d') == '1d'
                                and float(getattr(args, 'gabor_reg', 0.0)) > 0):
                            dual_gabor_reg = logits.new_tensor(0.0)
                            reg_count = 0
                            for branch in (self.model.vib_branch, self.model.aco_branch):
                                funconv = branch.funconv
                                if hasattr(funconv, 'superparams_init'):
                                    sp = funconv.superparams
                                    init_sp = funconv.superparams_init
                                    # f0 最重要；sigma 次之；2D theta 只做很弱约束。
                                    r = (sp[..., 0] - init_sp[..., 0]).pow(2).mean()
                                    if sp.shape[-1] > 1:
                                        r = r + 0.25 * (sp[..., 1] - init_sp[..., 1]).pow(2).mean()
                                    if sp.shape[-1] > 2:
                                        r = r + 0.01 * (sp[..., 2] - init_sp[..., 2]).pow(2).mean()
                                    dual_gabor_reg = dual_gabor_reg + r
                                    reg_count += 1
                            if reg_count > 0:
                                loss = loss + float(args.gabor_reg) * (dual_gabor_reg / reg_count)

                        #计算预测结果与正确数
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        if self.is_dual:
                            epoch_neural_correct += correct
                        # 损失记录（辅助损失时仅记录主损失，保证尺度一致）
                        if self.is_dual:
                            # Record the final classifier CE; branch auxiliary losses are optimization terms.
                            loss_for_record = self.criterion(logits, labels)
                        else:
                            loss_for_record = loss
                        loss_temp = loss_for_record.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # 最后一个epoch计算混淆矩阵
                        if epoch == args.max_epoch - 1:
                            for i, j in zip(labels.detach().cpu().numpy(), pred.detach().cpu().numpy()):
                                self.c_matrix[phase][i][j] += 1

                        # 训练阶段：反向传播+参数更新
                        if phase == 'train':

    # --------------------------------------------------------
    # 1. CNN梯度清零
    # --------------------------------------------------------
                           self.optimizer.zero_grad()

    # --------------------------------------------------------
    # 2. Gabor梯度清零
    # --------------------------------------------------------
                           if (args.model_name == 'TFN_Gabor'
                               and getattr(self, 'gabor_optimizer', None) is not None):

                               self.gabor_optimizer.zero_grad()


    # --------------------------------------------------------
    # 3. backward之前检查loss
    #
    # 如果forward已经产生NaN，
    # 在这里立即停止，而不是继续污染模型。
    # --------------------------------------------------------
                           if not torch.isfinite(loss):

                              raise FloatingPointError( "Non-finite loss BEFORE backward: "
            f"epoch={epoch}, "
            f"batch={batch_idx}, "
            f"loss={loss.item()}" )
    # --------------------------------------------------------
    # 4. 反向传播
    # --------------------------------------------------------
                           loss.backward()

                           # ============================================================
                           # DualModal stability guard
                           # Clean SNR=None uses the more sensitive 2D-Gabor front-end.
                           # Keep data/splits/labels untouched; only limit rare optimizer spikes.
                           # ============================================================
                           if self.is_dual:
                               dual_clip = float(getattr(args, 'dual_grad_clip', 5.0))
                               if dual_clip > 0.0:
                                   torch.nn.utils.clip_grad_norm_(
                                       self.model.parameters(), max_norm=dual_clip
                                   )

                               if getattr(self.model, 'gabor_mode', '1d') == '2d':
                                   gabor_params = getattr(
                                       self, '_dual_gabor_params',
                                       [
                                           self.model.vib_branch.funconv.superparams,
                                           self.model.aco_branch.funconv.superparams,
                                       ]
                                   )

                                   for gp in gabor_params:
                                       if gp.grad is not None and not torch.isfinite(gp.grad).all():
                                           bad_num = (~torch.isfinite(gp.grad)).sum().item()
                                           logging.warning(
                                               '[DualModal-2D-Gabor] non-finite gradient '
                                               'epoch=%d batch=%d bad=%d; replacing with 0',
                                               epoch, batch_idx, bad_num
                                           )
                                           torch.nan_to_num_(gp.grad, nan=0.0, posinf=0.0, neginf=0.0)

                                   gabor_clip = float(getattr(args, 'dual_gabor_grad_clip', 1.0))
                                   if gabor_clip > 0.0:
                                       torch.nn.utils.clip_grad_norm_(
                                           gabor_params, max_norm=gabor_clip
                                       )
    # ========================================================
    # 5. Gabor梯度处理
    # 仅在warmup结束以后执行
    # ========================================================
                           if (args.model_name == 'TFN_Gabor'
                               and epoch >= args.gabor_warmup_epochs ):
                               sp = self.model.funconv.superparams
                               grad = sp.grad
        # ----------------------------------------------------
        # 没有梯度
        # ----------------------------------------------------
                               if grad is None:
                                  logging.warning("[Gabor] superparams.grad is None | "
                                                  "epoch=%d batch=%d",
                                                  epoch,
                                                  batch_idx )
                               else:
            # ------------------------------------------------
            # A. 检测NaN / Inf梯度
            # ------------------------------------------------
                                    if not torch.isfinite(grad).all():
                                       bad_num = (~torch.isfinite(grad)).sum().item()
                                       logging.warning(
                                           "[Gabor] non-finite gradient detected | " 
                                           "epoch=%d batch=%d bad=%d",
                                           epoch,
                                           batch_idx,
                                           bad_num)
                # --------------------------------------------
                # 将NaN/Inf梯度变为0
                # --------------------------------------------
                                       torch.nan_to_num_(
                                           grad,
                                           nan=0.0,
                                           posinf=0.0,neginf=0.0)
            # ------------------------------------------------
            # B. 只允许f0学习
            # superparams[...,0] = f0
            # superparams[...,1] = sigma
            # superparams[...,2] = theta
            # -----------------------------------------------
            # sigma梯度强制为0
                                    grad[..., 1].zero_()
            # theta在1D Gabor中没有参与核生成，因此固定
                                    if grad.shape[-1] > 2:
                                       grad[..., 2].zero_()
            # ------------------------------------------------
            # C. f0逐元素梯度裁剪
            # 不再使用clip_grad_norm_
            # ------------------------------------------------
                                    grad[..., 0].clamp_(min=-0.01, max=0.01 )
            # ------------------------------------------------
            # D. 每个epoch第一个batch打印梯度
            # ------------------------------------------------
                                    if batch_idx == 0:
                                       f0_grad = grad[..., 0]
                                       logging.info("[Gabor-grad] epoch=%d | "
                                                    "finite=%s | "
                                                    "mean_abs=%.8e | "
                                                    "max_abs=%.8e",
                                                    epoch,
                                                    str(torch.isfinite(f0_grad).all().item()),
                                                    f0_grad.abs().mean().item(),
                                                    f0_grad.abs().max().item())
    # 6. 更新CNN参数
                           self.optimizer.step()
                           if self.is_dual and self.use_dual_ema:
                               self._update_dual_ema(epoch)
    # 7. 更新Gabor参数
    # 前warmup阶段不执行；
    # warmup结束后才执行SGD。
                           if (args.model_name == 'TFN_Gabor'
                               and epoch >= args.gabor_warmup_epochs
                               and getattr(self,'gabor_optimizer',None) is not None):
                             self.gabor_optimizer.step()
        # 8. optimizer.step以后做参数安全检查
                             with torch.no_grad():
                                sp = (self.model.funconv
                                      .superparams)
                                init_sp = (self.model.funconv
                                           .superparams_init)
            # A. 检查Gabor参数是否出现NaN / Inf
                                bad = ~torch.isfinite(sp)
                                if bad.any():
                                   bad_num = bad.sum().item()
                                   logging.warning(
                                       "[Gabor] non-finite parameter "
                                        "AFTER optimizer.step | "
                                        "bad=%d, restoring initial values",
                                        bad_num)
                # 非有限参数恢复为初始值
                                   sp[bad] = init_sp[bad]
            # B. 限制f0物理范围
                                f0_low = (self.model.funconv
                                          .params_bound[0][0])
                                f0_high =(self.model.funconv
                                          .params_bound[0][1])
                                sp[..., 0].clamp_(
                                    f0_low,
                                    f0_high)
            # C. sigma强制恢复到初始化值
            # 即sigma始终保持0.5
                                sp[..., 1].copy_(
                                      init_sp[..., 1])
            # D. theta固定为初始值
                                if sp.shape[-1] > 2:
                                    sp[..., 2].copy_(
                                        init_sp[..., 2])
                            #批次记录
                        batch_loss += loss_temp
                        batch_acc += correct
                        batch_count += inputs.size(0)

                            #打印训练信息
                        if (batch_idx + 1) % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info(
                                    '<tempinfo> Epoch: {:3d} [{:4d}/{:4d}],  step: {:3d},  Train Loss: {:.3f},  Train Acc: {:.2f},  {:8.1f} samples/sec,  {:.4f} sec/batch'
                                    .format(epoch, (batch_idx + 1) * len(inputs), len(self.dataloaders[phase].dataset),
                                            batch_idx + 1, batch_loss, batch_acc * 100, sample_per_sec, batch_time))
                                self.MinorRecords['train_loss'].append(batch_loss)
                                self.MinorRecords['train_acc'].append(batch_acc)
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0

                # 在每个训练周期结束时打印周期信息（包括训练集与验证集）
                # 根据数据集大小归一化（防止数值为零）
                ds_len = len(self.datasets[phase]) if phase in self.datasets and self.datasets[phase] is not None else 0
                if ds_len > 0:
                    epoch_loss = epoch_loss / ds_len
                    epoch_acc = epoch_acc / ds_len
                else:
                    epoch_loss = 0.0
                    epoch_acc = 0.0
                epoch_neural_acc = epoch_acc
                live_val_acc = None
                if self.is_dual and phase == 'val':
                    live_val_acc = (epoch_live_correct / ds_len) if ds_len > 0 else 0.0
                suffix = ' [pure DualModal_TFN, EMA]' if (self.is_dual and phase == 'val' and self.use_dual_ema and getattr(self, 'ema_started', False)) else ''
                logging.info('<info> Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec{}'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc * 100, time.time() - epoch_start, suffix
                ))
                if self.is_dual and phase == 'val' and self.use_internal_spectral and ds_len > 0:
                    logging.info('<info> Epoch: %d val-SpectralOnly-Acc: %.4f (internal DualModal_TFN branch)',
                                 epoch, 100.0 * epoch_spectral_correct / ds_len)
                if self.is_dual and phase == 'val' and live_val_acc is not None and self.use_dual_ema and getattr(self, 'ema_started', False):
                    logging.info('<info> Epoch: %d val-Live-Acc: %.4f (pure DualModal_TFN; selection compares Live vs EMA)',
                                 epoch, live_val_acc * 100.0)
                if phase == 'train' and self.is_dual and self.use_internal_spectral and ds_len > 0:
                    alpha_val = float(self.model.spectral_fusion_alpha().detach().cpu().item()) if hasattr(self.model, 'spectral_fusion_alpha') else 0.0
                    logging.info('<info> Epoch: %d train-SpectralOnly-Acc: %.4f, spectral-fusion-alpha: %.4f (INTERNAL model branch)',
                                 epoch, 100.0 * epoch_spectral_correct / ds_len, alpha_val)

                # 记录轮次信息 epoch级损失/精度
                self.Records["%s_loss" % phase].append(epoch_loss)
                self.Records["%s_acc" % phase].append(epoch_acc)

                # 验证阶段：保存最优模型及最终模型（最优模型基于验证集选取）
                if phase == 'val':
                    selection_acc = epoch_acc
                    state_source = forward_model
                    selection_tag = 'EMA' if (self.is_dual and self.use_dual_ema and getattr(self, 'ema_started', False)) else 'LIVE'
                    if (self.is_dual and self.use_dual_ema and getattr(self, 'ema_started', False)
                            and live_val_acc is not None and live_val_acc > selection_acc):
                        selection_acc = live_val_acc
                        state_source = self.model
                        selection_tag = 'LIVE'
                    if selection_acc > best_acc:
                        best_acc = selection_acc
                        self.Records['best_epoch'] = epoch
                        logging.info(
                            "save best epoch %d, PURE DualModal validation acc %.4f [%s]",
                            epoch, selection_acc, selection_tag)
                        save_best_data_dir = os.path.join(self.save_dir,
                                                          'epoch{}-acc{:.4f}-best_model.pth'.format(epoch,
                                                                                                    selection_acc * 100))
                        save_best_data = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(state_source.state_dict()),
                                          'optimizer': copy.deepcopy(self.optimizer.state_dict()), 'opt': self.args.opt,
                                          'pure_dualmodal': True, 'ema': bool(self.is_dual and self.use_dual_ema)}
                        # 在训练器实例上保留最佳参考数据
                        self.save_best_data = save_best_data
                        self.save_best_data_dir = save_best_data_dir

                    if epoch == args.max_epoch - 1:
                        if args.save_model:
                            if save_best_data is not None:
                                torch.save(save_best_data, save_best_data_dir)
                            logging.info("save final epoch {}, final acc {:.4f}".format(epoch, epoch_acc))
                            save_data = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                         'optimizer': self.optimizer.state_dict(), 'opt': self.args.opt}
                            torch.save(save_data,
                                       os.path.join(self.save_dir,
                                                    'epoch{}-acc{:.4f}-final_model.pth'.format(epoch, epoch_acc * 100)))
            # =====================================================
            # 每个epoch训练+验证结束以后：
            # 监控Gabor参数变化
            # =====================================================
            if args.model_name == 'TFN_Gabor':

                with torch.no_grad():

                    sp = self.model.funconv.superparams.detach()

                    f0 = sp[..., 0]
                    sigma = sp[..., 1]

                    # ---------------------------------------------
                    # 如果TFconvlayer.py中已经保存了初始化参数
                    # ---------------------------------------------
                    if hasattr(
                        self.model.funconv,
                        'superparams_init'
                    ):

                        sp_init = (
                            self.model.funconv
                            .superparams_init
                        )

                        f0_init = sp_init[..., 0]
                        sigma_init = sp_init[..., 1]

                        # 相对于初始化值的平均绝对漂移
                        f0_drift = (
                            f0 - f0_init
                        ).abs().mean().item()

                        sigma_drift = (
                            sigma - sigma_init
                        ).abs().mean().item()

                    else:

                        f0_drift = float('nan')
                        sigma_drift = float('nan')

                    logging.info(
                        "[Gabor] epoch=%d | "
                        "f0[min=%.6f max=%.6f "
                        "mean=%.6f drift=%.8f] | "
                        "sigma[min=%.6f max=%.6f "
                        "mean=%.6f drift=%.8f]",
                        epoch,

                        f0.min().item(),
                        f0.max().item(),
                        f0.mean().item(),
                        f0_drift,

                        sigma.min().item(),
                        sigma.max().item(),
                        sigma.mean().item(),
                        sigma_drift
                    )
            # 每个迭代周期更新学习率
            try:
                # 每个批次已完成优化器迭代；若配置了学习率调度器，则在此处执行调度器迭代
                if getattr(self, 'lr_scheduler', None) is not None:
                    self.lr_scheduler.step()
            except Exception:
                logging.exception('Failed to step lr_scheduler')

        # 训练流程完成后，将记录列表堆叠转换为数组（保存训练记录）
        for k in self.Records.keys():
            self.Records[k] = np.array(self.Records[k])
        for k in self.MinorRecords.keys():
            self.MinorRecords[k] = np.array(self.MinorRecords[k])
        # 保存记录
        io.savemat(os.path.join(self.save_dir, "Records.mat"), self.Records)
        io.savemat(os.path.join(self.save_dir, "MinorRecords.mat"), self.MinorRecords)

        # 开展最终测试评估 
        # 在测试集上进行评估（如有最优模型则使用）
        try:
            if hasattr(self, 'dataloaders') and 'test' in self.dataloaders and self.dataloaders['test'] is not None and hasattr(self.dataloaders['test'], 'dataset') and len(self.dataloaders['test'].dataset) > 0:
                logging.info('Running final evaluation on test set (v5: one DualModal_TFN model; no external teacher/KD)')
                # 如有可用最优模型则临时加载（备份当前模型状态）
                backup_state = None
                if getattr(self, 'save_best_data', None) is not None:
                    try:
                        backup_state = copy.deepcopy(self.model.state_dict())
                        self.model.load_state_dict(self.save_best_data['state_dict'])
                    except Exception:
                        logging.exception('Failed to load save_best_data for test evaluation; using current model')

                self.model.eval()
                total_loss = 0.0
                total_correct = 0
                total_n = 0
                all_test_labels = []
                all_test_preds = []
                with torch.no_grad(): #禁用梯度
                    for batch_idx, (inputs, labels) in enumerate(self.dataloaders['test']):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        if self.is_dual and self.dual_balanced_snr:
                            inputs = self._apply_dual_balanced_awgn(
                                inputs, self._requested_val_snr, 'test', 0, batch_idx)
                        #前向传播
                        if self.is_dual:
                            x_vib, x_aco = self._split_inputs(inputs)
                            # v5 FINAL TEST = one DualModal_TFN model, including its internal learnable spectral branch.
                            logits = self.model(x_vib, x_aco)
                        else:
                            inputs_single = self._split_inputs(inputs)
                            logits = self.model(inputs_single)
                        loss = self.criterion(logits, labels)
                        batch_n = inputs.size(0)
                        total_loss += loss.item() * batch_n
                        preds = logits.argmax(dim=1)
                        all_test_labels.extend(labels.detach().cpu().numpy().tolist())
                        all_test_preds.extend(preds.detach().cpu().numpy().tolist())
                        total_correct += torch.eq(preds, labels).float().sum().item()
                        total_n += batch_n
                        # 累积混淆矩阵（用于最终的 precision/recall/f1 计算）
                        try:
                            for i, j in zip(labels.detach().cpu().numpy(), preds.detach().cpu().numpy()):
                                self.c_matrix['test'][int(i)][int(j)] += 1
                        except Exception:
                            logging.exception('Failed to update test confusion matrix')
                #计算测试集损失/精度
                    if total_n > 0:
                        test_loss = total_loss / total_n
                        test_acc = total_correct / total_n
                        logging.info('<test> Test-Loss: {:.4f} Test-Acc: {:.4f}'.format(test_loss, test_acc * 100))
                        # 计算 Precision / Recall / F1（macro average）
                        try:
                            cm = self.c_matrix['test'].astype(float)
                            tp = np.diag(cm)
                            fp = cm.sum(axis=0) - tp
                            fn = cm.sum(axis=1) - tp
                            eps = 1e-12
                            precision_per = tp / (tp + fp + eps)
                            recall_per = tp / (tp + fn + eps)
                            f1_per = 2 * precision_per * recall_per / (precision_per + recall_per + eps)
                            precision_macro = float(np.mean(precision_per))
                            recall_macro = float(np.mean(recall_per))
                            f1_macro = float(np.mean(f1_per))
                            logging.info('<test> Test-Precision: {:.4f} Test-Recall: {:.4f} Test-F1: {:.4f}'.format(
                                precision_macro * 100, recall_macro * 100, f1_macro * 100))
                        except Exception:
                            logging.exception('Failed to compute Precision/Recall/F1 for test set')
                    # 将最终测试指标追加至记录中
                    try:
                        self.Records['test_loss'] = np.append(self.Records.get('test_loss', np.array([])), test_loss)
                        self.Records['test_acc'] = np.append(self.Records.get('test_acc', np.array([])), test_acc)
                        # 追加 precision/recall/f1 到记录（以 macro 值保存）
                        try:
                            self.Records['test_precision'] = np.append(self.Records.get('test_precision', np.array([])), precision_macro)
                            self.Records['test_recall'] = np.append(self.Records.get('test_recall', np.array([])), recall_macro)
                            self.Records['test_f1'] = np.append(self.Records.get('test_f1', np.array([])), f1_macro)
                        except Exception:
                            # 如果上面计算失败，则确保键存在（防止后续访问错误）
                            self.Records['test_precision'] = np.array([precision_macro]) if 'precision_macro' in locals() else np.array([])
                            self.Records['test_recall'] = np.array([recall_macro]) if 'recall_macro' in locals() else np.array([])
                            self.Records['test_f1'] = np.array([f1_macro]) if 'f1_macro' in locals() else np.array([])
                    except Exception:
                        # 备用方案：确保键存在
                        self.Records['test_loss'] = np.array([test_loss])
                        self.Records['test_acc'] = np.array([test_acc])
                    np.save(os.path.join(self.save_dir,"test_labels.npy"),
                            np.asarray(all_test_labels,dtype=np.int64))
                    np.save(os.path.join(self.save_dir,"test_preds.npy"),
                            np.asarray(all_test_preds,dtype=np.int64))
                    logging.info("Final test confusion matrix:\n%s",
                                 self.c_matrix['test'].astype(int))
                    logging.info("Saved test_labels.npy and test_preds.npy, N=%d",len(all_test_labels))    

                # 若存在备份则恢复模型状态
                if backup_state is not None:
                    try:
                        self.model.load_state_dict(backup_state)
                    except Exception:
                        logging.exception('Failed to restore original model state after test evaluation')
        except Exception:
            logging.exception('Final test evaluation failed')

        # 如果实例中已存在save_best_data，则保留该数据（已完成存储）；否则从本地进行赋值。
        if getattr(self, 'save_best_data', None) is None and save_best_data is not None:
            self.save_best_data = save_best_data
            self.save_best_data_dir = save_best_data_dir

        # 重要：最终 test 指标是在第一次 savemat 之后才追加的，因此测试完成后必须再次保存。
        # 否则外部汇总脚本读取 Records.mat 时会看不到 test_acc / Macro-P/R/F1。
        try:
            io.savemat(os.path.join(self.save_dir, "Records.mat"), self.Records)
        except Exception:
            logging.exception("Failed to resave Records.mat after final test evaluation")

        # 记录最优且最终的准确率与损失值，最终准确率通常取最后5个训练轮次的平均值
        final_len = int(max(min(args.max_epoch * 0.5, 5), 1))
        #若未生成任何记录，则输出简短提示信息，不执行argmax/均值计算
        if self.Records.get('train_acc', None) is None or len(self.Records['train_acc']) == 0:
            info = "No training/validation/test records were produced. Datasets may be empty.\n"
        else:
            info = "max train acc in epoch {:2d}: {:10.6f}\n".format(int(self.Records['train_acc'].argmax()) + 1,
                                         float(self.Records['train_acc'].max())) \
                   + "max val acc in epoch {:2d}: {:10.6f}\n".format(int(self.Records['val_acc'].argmax()) + 1 if len(self.Records['val_acc'])>0 else 0,
                                         float(self.Records['val_acc'].max()) if len(self.Records['val_acc'])>0 else 0.0) \
                   + "max test acc in epoch {:2d}: {:10.6f}\n".format(int(self.Records['test_acc'].argmax()) + 1 if len(self.Records['test_acc'])>0 else 0,
                                         float(self.Records['test_acc'].max()) if len(self.Records['test_acc'])>0 else 0.0) \
                   + "final train acc: %.6f\n final val acc: %.6f\n final test acc: %.6f\n" \
                   % (float(self.Records['train_acc'][-final_len:].mean()), float(self.Records['val_acc'][-final_len:].mean()) if len(self.Records['val_acc'])>0 else 0.0,
                  (float(self.Records['test_acc'][-final_len:].mean()) if len(self.Records['test_acc'])>0 else 0.0))
        # 如有计算到 precision/recall/f1，则附加到输出信息中（取最后 final_len 轮的均值）
        try:
            if 'test_precision' in self.Records and len(self.Records['test_precision']) > 0:
                final_precision = float(self.Records['test_precision'][-final_len:].mean())
                final_recall = float(self.Records['test_recall'][-final_len:].mean())
                final_f1 = float(self.Records['test_f1'][-final_len:].mean())
                info += "final test precision: %.6f\n final test recall: %.6f\n final test f1: %.6f\n" % (
                    final_precision, final_recall, final_f1)
        except Exception:
            logging.exception('Failed to compute final precision/recall/f1 for info output')

        for item in info.split('\n'):
            logging.info(item)
        with open(os.path.join(self.save_dir, 'acc output.txt'), 'w', encoding='utf-8') as f:
            f.write(info)

    def plot_save(self):
        """
        训练完成后绘制混淆矩阵与损失曲线
        """
        # 设置颜色
        current_cmap = sns.color_palette("husl", 10)
        sns.set(style="white")
        #优先选用支持中文字形的字体；备选通用字体
        try:
            import matplotlib.font_manager as fm # pyright: ignore[reportMissingModuleSource]
            # 通用的支持中文的Windows/类Unix系统字体
            preferred = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
            available = [f.name for f in fm.fontManager.ttflist]
            chosen = None
            for name in preferred:
                if name in available:
                    chosen = name
                    break
            if chosen is not None:
                mplb.rcParams['font.sans-serif'] = [chosen]
                mplb.rcParams['font.family'] = 'sans-serif'
        except Exception:
            pass
        mplb.rcParams['axes.unicode_minus'] = False #负号显示正常
        sns.set(style="ticks", context="notebook", palette=current_cmap, font_scale=1.5)

        # 创建可视化保存目录
        self.save_dir_sub = os.path.join(self.save_dir, "postprosess")
        if not os.path.exists(self.save_dir_sub):
            os.makedirs(self.save_dir_sub)

        # 绘制混淆矩阵
        mplb.rcParams['font.size'] = 12
        for phase in ['train', 'val', 'test']:
            f, ax = plt.subplots(figsize=(10, 8), dpi=100)
            sns.heatmap(self.c_matrix[phase], annot=True, ax=ax) #热力图+数值标注
            ax.invert_yaxis() #Y轴反转（真实标签从上到下）
            ax.set_xticklabels(self.label_name) #X轴：预测标签
            ax.set_yticklabels(self.label_name) #Y轴：真实标签
            ax.set_xlabel('predict', fontsize=15)
            ax.set_ylabel('true', fontsize=15)
            ax.set_title('confusion matrix: %s' % phase, fontsize=18)
            f.savefig(os.path.join(self.save_dir_sub, "confusion_matrix_%s.jpg" % phase))

        # ============================================================
        # 论文最终版训练收敛曲线（仅修改绘图，不改变训练/模型/验证/测试）
        #
        # 关键原则：
        # 1) 不再使用 MinorRecords。MinorRecords 是 print_step 级记录，
        #    且原实现会把不同 phase 的中间统计混入同一序列，容易产生
        #    “训练精度剧烈下探”的视觉假象。
        # 2) 直接使用 Records 中每个 epoch 结束后的真实训练集平均值：
        #       Records['train_loss']
        #       Records['train_acc']
        # 3) 主图放大 0-25 epoch；右上角 inset 展示完整 0-100 epoch。
        # 4) 不做平滑、不删异常点、不改任何数值。
        # ============================================================
        train_loss_ep = np.asarray(self.Records['train_loss'], dtype=float).reshape(-1)
        train_acc_ep = np.asarray(self.Records['train_acc'], dtype=float).reshape(-1) * 100.0

        n_epoch = min(train_loss_ep.size, train_acc_ep.size)

        if n_epoch > 0:
            train_loss_ep = train_loss_ep[:n_epoch]
            train_acc_ep = train_acc_ep[:n_epoch]
            epochs = np.arange(1, n_epoch + 1, dtype=float)

            # 主图重点展示前25个epoch；完整训练过程放在inset。
            focus_end = min(25.0, float(n_epoch))
            full_end = float(n_epoch)
            focus_mask = epochs <= focus_end

            # 论文常用的简洁两联图。
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8.0, 7.0), dpi=150, sharex=True
            )

            # ---------------- Loss: 0-25 epoch ----------------
            loss_line, = ax1.plot(
                epochs,
                train_loss_ep,
                linewidth=2.0,
                marker='o',
                markersize=3.2,
                markevery=1,
                label='Train',
                zorder=3
            )
            ax1.set_ylabel('Cross-entropy loss')
            ax1.set_xlim(0.0, focus_end)

            # 主图loss范围只由前25个epoch的真实epoch均值决定，
            # 避免100个epoch的横向压缩影响早期收敛可读性。
            focus_loss = train_loss_ep[focus_mask]
            focus_loss = focus_loss[np.isfinite(focus_loss)]
            if focus_loss.size:
                loss_max = float(np.max(focus_loss))
                ax1.set_ylim(0.0, max(0.05, loss_max * 1.08))

            ax1.grid(True, alpha=0.18, linewidth=0.6)
            ax1.legend(frameon=False, fontsize=9, loc='upper right')

            # 完整0-100 epoch loss inset
            ax1_in = ax1.inset_axes([0.62, 0.43, 0.35, 0.48])
            ax1_in.plot(
                epochs,
                train_loss_ep,
                linewidth=1.15,
                color=loss_line.get_color()
            )
            ax1_in.set_xlim(0.0, full_end)
            full_loss = train_loss_ep[np.isfinite(train_loss_ep)]
            if full_loss.size:
                full_loss_max = float(np.max(full_loss))
                ax1_in.set_ylim(0.0, max(0.05, full_loss_max * 1.08))
            ax1_in.grid(True, alpha=0.15, linewidth=0.45)
            ax1_in.tick_params(labelsize=7, direction='out', length=2.5)
            ax1_in.set_title(
                '0-{} epochs'.format(int(full_end)),
                fontsize=8,
                pad=2
            )
            if full_end >= 100:
                ax1_in.set_xticks([0, 50, 100])
            else:
                ax1_in.set_xticks([0, full_end / 2.0, full_end])

            # ---------------- Accuracy: 0-25 epoch ----------------
            acc_line, = ax2.plot(
                epochs,
                train_acc_ep,
                linewidth=2.0,
                marker='o',
                markersize=3.2,
                markevery=1,
                label='Train',
                zorder=3
            )
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_xlim(0.0, focus_end)

            # 不再固定为[80,100]。根据前25个epoch真实Train Acc自动确定，
            # 既能展示BJTU早期提升，又不会把低值阶段截掉。
            focus_acc = train_acc_ep[focus_mask]
            focus_acc = focus_acc[np.isfinite(focus_acc)]
            if focus_acc.size:
                acc_min = float(np.min(focus_acc))
                lower = max(0.0, 5.0 * np.floor((acc_min - 5.0) / 5.0))
                ax2.set_ylim(lower, 100.5)
            else:
                ax2.set_ylim(0.0, 100.5)

            ax2.grid(True, alpha=0.18, linewidth=0.6)
            ax2.legend(frameon=False, fontsize=9, loc='lower right')

            # 完整0-100 epoch accuracy inset
            ax2_in = ax2.inset_axes([0.62, 0.10, 0.35, 0.43])
            ax2_in.plot(
                epochs,
                train_acc_ep,
                linewidth=1.15,
                color=acc_line.get_color()
            )
            ax2_in.set_xlim(0.0, full_end)

            full_acc = train_acc_ep[np.isfinite(train_acc_ep)]
            if full_acc.size:
                full_acc_min = float(np.min(full_acc))
                full_lower = max(
                    0.0,
                    5.0 * np.floor((full_acc_min - 5.0) / 5.0)
                )
                ax2_in.set_ylim(full_lower, 100.5)
            else:
                ax2_in.set_ylim(0.0, 100.5)

            ax2_in.grid(True, alpha=0.15, linewidth=0.45)
            ax2_in.tick_params(labelsize=7, direction='out', length=2.5)
            ax2_in.set_title(
                '0-{} epochs'.format(int(full_end)),
                fontsize=8,
                pad=2
            )
            if full_end >= 100:
                ax2_in.set_xticks([0, 50, 100])
            else:
                ax2_in.set_xticks([0, full_end / 2.0, full_end])

            # 主图刻度与边框学术化处理。
            ax2.set_xticks(np.arange(0, focus_end + 0.1, 5))
            for ax in (ax1, ax2):
                ax.tick_params(direction='out')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            fig.tight_layout()

            # 保留旧文件名，直接兼容你原来的工作流。
            fig.savefig(
                os.path.join(self.save_dir_sub, "Minor_loss_acc.jpg"),
                dpi=400,
                bbox_inches='tight'
            )
            # 同时保存论文推荐的高分辨率PNG和矢量PDF。
            fig.savefig(
                os.path.join(self.save_dir_sub, "Train_epoch_convergence.png"),
                dpi=600,
                bbox_inches='tight'
            )
            fig.savefig(
                os.path.join(self.save_dir_sub, "Train_epoch_convergence.pdf"),
                bbox_inches='tight'
            )
            plt.close(fig)

            # 导出本图实际使用的epoch级数据，便于论文复核/Origin重绘。
            try:
                import csv as _csv
                with open(
                    os.path.join(self.save_dir_sub, "Train_epoch_convergence.csv"),
                    'w',
                    newline='',
                    encoding='utf-8'
                ) as _f:
                    _writer = _csv.writer(_f)
                    _writer.writerow([
                        'epoch',
                        'train_loss_epoch_mean',
                        'train_acc_epoch_percent'
                    ])
                    for _i in range(n_epoch):
                        _writer.writerow([
                            _i + 1,
                            float(train_loss_ep[_i]),
                            float(train_acc_ep[_i])
                        ])
            except Exception:
                logging.exception(
                    'Failed to save Train_epoch_convergence.csv'
                )

        # -----------------Gabor分析与图像绘制-----------------
        try:
            # 确定要检查的状态：优先选用已保存状态，无则选用最终状态
            if getattr(self, 'save_best_data', None) is not None:
                state = self.save_best_data['state_dict']
                # 临时加载最优状态
                model_backup = self.model.state_dict()
                try:
                    self.model.load_state_dict(state)
                except Exception:
                    #若加载失败，则回退至当前模型
                    logging.exception('Failed to load best state_dict, using current model params')
                    self.model.load_state_dict(model_backup)
            else:
                model_backup = None

            # 从模型中提取Gabor超参数
            gabor_params = None
            kernels = None
            # 单模态TFN拥有getweight()函数
            if hasattr(self.model, 'getweight'):
                try:
                    kernels, gabor_params = self.model.getweight()
                except Exception:
                    logging.exception('model.getweight() failed')
            else:
                # 双模态时域融合网络（DualModal_TFN）或其他模型：提取振动分支/声学分支的Gabor参数
                try:
                    params_v = None
                    params_a = None
                    kern_v = None
                    kern_a = None
                    if hasattr(self.model, 'vib_branch'):
                        vb = self.model.vib_branch
                        if hasattr(vb, 'funconv'):
                            # TFconv 可能会输出`权重`（传统版TFconv_Gabor）或`superparams`（TFconv_Gabor2D），两种情况均做处理。
                            if hasattr(vb.funconv, 'weight') and getattr(vb.funconv, 'weight') is not None:
                                kern_v = vb.funconv.weight.cpu().detach().numpy()
                                params_v = None
                                try:
                                    params_v = vb.funconv.superparams.cpu().detach().numpy()
                                except Exception:
                                    params_v = None
                            elif hasattr(vb.funconv, 'superparams') and getattr(vb.funconv, 'superparams') is not None:
                                params_v = vb.funconv.superparams.cpu().detach().numpy()
                                # 通过压缩频率轴，由二维Gabor参数重构一维实部/虚部卷积核
                                try:
                                    sp = params_v
                                    out_c, in_c, p = sp.shape
                                    ksize = int(getattr(vb.funconv, 'kernel_size', getattr(vb.funconv, 'funckernel_size', 15)))
                                    half = ksize // 2
                                    coords = np.linspace(-half, half, ksize)
                                    Y, X = np.meshgrid(coords, coords)
                                    real_k = np.zeros((out_c, in_c, ksize), dtype=np.float32)
                                    imag_k = np.zeros((out_c, in_c, ksize), dtype=np.float32)
                                    for i in range(out_c):
                                        for j in range(in_c):
                                            f0 = float(sp[i, j, 0])
                                            sigma = float(sp[i, j, 1])
                                            theta = float(sp[i, j, 2]) if p > 2 else 0.0
                                            c = np.cos(theta); s = np.sin(theta)
                                            Xp = X * c + Y * s
                                            Yp = -X * s + Y * c
                                            denom = max(sigma * half, 1e-6)
                                            gauss = np.exp(-(Xp**2 + Yp**2) / (2.0 * denom**2))
                                            real2d = gauss * np.cos(2 * math.pi * f0 * Xp)
                                            imag2d = gauss * np.sin(2 * math.pi * f0 * Xp)
                                            # 折叠频率轴（行维度）以得到时间核
                                            real_k[i, j, :] = real2d.mean(axis=0)
                                            imag_k[i, j, :] = imag2d.mean(axis=0)
                                    # 匹配旧版内核排布格式：(2, 输出通道, 输入通道, 卷积核尺寸)
                                    kern_v = np.stack([real_k, imag_k], axis=0)
                                except Exception:
                                    logging.exception('Failed to reconstruct 1D kernels from TFconv_Gabor2D superparams for vib_branch')
                                    kern_v = None
                            else:
                                kern_v = None
                    #声音分支与振动分支一致
                    if hasattr(self.model, 'aco_branch'):
                        #逻辑同vib_branch，提取kern_a/params_a
                        ab = self.model.aco_branch
                        if hasattr(ab, 'funconv'):
                            if hasattr(ab.funconv, 'weight') and getattr(ab.funconv, 'weight') is not None:
                                kern_a = ab.funconv.weight.cpu().detach().numpy()
                                params_a = None
                                try:
                                    params_a = ab.funconv.superparams.cpu().detach().numpy()
                                except Exception:
                                    params_a = None
                            elif hasattr(ab.funconv, 'superparams') and getattr(ab.funconv, 'superparams') is not None:
                                params_a = ab.funconv.superparams.cpu().detach().numpy()
                                try:
                                    sp = params_a
                                    out_c, in_c, p = sp.shape
                                    ksize = int(getattr(ab.funconv, 'kernel_size', getattr(ab.funconv, 'funckernel_size', 15)))
                                    half = ksize // 2
                                    coords = np.linspace(-half, half, ksize)
                                    Y, X = np.meshgrid(coords, coords)
                                    real_k = np.zeros((out_c, in_c, ksize), dtype=np.float32)
                                    imag_k = np.zeros((out_c, in_c, ksize), dtype=np.float32)
                                    for i in range(out_c):
                                        for j in range(in_c):
                                            f0 = float(sp[i, j, 0])
                                            sigma = float(sp[i, j, 1])
                                            theta = float(sp[i, j, 2]) if p > 2 else 0.0
                                            c = np.cos(theta); s = np.sin(theta)
                                            Xp = X * c + Y * s
                                            Yp = -X * s + Y * c
                                            denom = max(sigma * half, 1e-6)
                                            gauss = np.exp(-(Xp**2 + Yp**2) / (2.0 * denom**2))
                                            real2d = gauss * np.cos(2 * math.pi * f0 * Xp)
                                            imag2d = gauss * np.sin(2 * math.pi * f0 * Xp)
                                            real_k[i, j, :] = real2d.mean(axis=0)
                                            imag_k[i, j, :] = imag2d.mean(axis=0)
                                    kern_a = np.stack([real_k, imag_k], axis=0)
                                except Exception:
                                    logging.exception('Failed to reconstruct 1D kernels from TFconv_Gabor2D superparams for aco_branch')
                                    kern_a = None
                            else:
                                kern_a = None
                    # 若存在vib则优先选用vib
                    if params_v is not None:
                        gabor_params = params_v
                        kernels = kern_v
                    else:
                        gabor_params = params_a
                        kernels = kern_a
                except Exception:
                    logging.exception('Failed to extract vib/aco funconv params')

            if gabor_params is not None:
                # Gabor参数维度：(输出通道数, 输入通道数, 参数p)
                sp = gabor_params.reshape(-1, gabor_params.shape[-1])
                f0 = sp[:, 0]
                sigma = sp[:, 1] if sp.shape[1] > 1 else np.zeros_like(f0)
                theta = sp[:, 2] if sp.shape[1] > 2 else None
                # 基频f0与标准差sigma的散点图
                f, ax = plt.subplots(figsize=(6, 4), dpi=100)
                ax.scatter(f0, sigma, c='C0', s=20)
                ax.set_xlabel('f0 (normalized)')
                ax.set_ylabel('sigma (normalized)')
                ax.set_title('Gabor learned parameters (f0 vs sigma)')
                f.savefig(os.path.join(self.save_dir_sub, 'gabor_params_scatter.jpg'))

                #将数值参数保存至文件（npz格式）
                try:
                    import numpy as _np 
                    if theta is None:
                        _np.savez_compressed(os.path.join(self.save_dir_sub, 'gabor_params.npz'), f0=f0, sigma=sigma)
                    else:
                        _np.savez_compressed(os.path.join(self.save_dir_sub, 'gabor_params.npz'), f0=f0, sigma=sigma, theta=theta)
                except Exception:
                    logging.exception('Failed to save gabor params npz')

                # 若存在卷积核，则绘制示例卷积核（实部+虚部）
                if kernels is not None:
                    # 内核尺寸预期：(2, 输出通道数, 输入通道数, 卷积核尺寸)
                    try:
                        kern = kernels
                        # 最多选择8个筛选条件
                        out_chan = kern.shape[1]
                        sel = list(range(min(8, out_chan)))
                        fig_k, axes = plt.subplots(len(sel), 2, figsize=(8, 2 * len(sel)), dpi=100)
                        if len(sel) == 1:
                            axes = axes.reshape(1, 2)
                        for idx_i, idx in enumerate(sel):
                            real = kern[0, idx, 0, :]
                            imag = kern[1, idx, 0, :]
                            axes[idx_i, 0].plot(real)
                            axes[idx_i, 0].set_title(f'filter {idx} real')
                            axes[idx_i, 1].plot(imag)
                            axes[idx_i, 1].set_title(f'filter {idx} imag')
                        fig_k.tight_layout()
                        fig_k.savefig(os.path.join(self.save_dir_sub, 'gabor_kernels_example.jpg'))
                    except Exception:
                        logging.exception('Failed to plot gabor kernels')

                # 针对少量验证样本生成回复
                try:
                    #从验证集中选取3个样本
                    num_samples = min(3, len(self.datasets['val']))
                    for si in range(num_samples):
                        sample = self.datasets['val'][si][0]  
                        x = sample
                        if isinstance(x, np.ndarray):
                            x = torch.tensor(x)
                        # 保证形状为 (1, C, L)
                        if x.ndim == 1:
                            x = x.unsqueeze(0).unsqueeze(0)
                        elif x.ndim == 2:
                            x = x.unsqueeze(0)
                        x = x.to(torch.float32)
                        # 在区间内选取4个筛选索引
                        if gabor_params.shape[0] >= 4:
                            sel_filters = [0, gabor_params.shape[0] // 4, gabor_params.shape[0] // 2, -1]
                        else:
                            sel_filters = list(range(gabor_params.shape[0]))

                        fig_r, axs_r = plt.subplots(len(sel_filters) + 1, 1, figsize=(8, 2 * (len(sel_filters) + 1)), dpi=100)
                        axs_r[0].plot(x.squeeze().cpu().numpy())
                        axs_r[0].set_title('original signal (val sample %d)' % si)
                        #使用每个选定的滤波器执行卷积计算
                        for ii, fid in enumerate(sel_filters):
                            try:
                                # 确保一维卷积层采用单通道输入（双通道场景下选用振动通道）
                                x_in = x
                                if x_in.ndim == 3 and x_in.shape[1] == 2:
                                    x_in = x_in[:, 0:1, :]

                                # 为该滤波器构建卷积权重：实部+虚部 → 包络
                                w_real_np = kernels[0, fid, 0, :].reshape(1, 1, -1)
                                w_imag_np = kernels[1, fid, 0, :].reshape(1, 1, -1)

                                # 将权重转换为与输入数据相同设备、相同数据类型的张量
                                w_real_t = torch.from_numpy(w_real_np).to(dtype=x_in.dtype, device=x_in.device)
                                w_imag_t = torch.from_numpy(w_imag_np).to(dtype=x_in.dtype, device=x_in.device)

                                xr = F.conv1d(x_in, w_real_t)
                                xi = F.conv1d(x_in, w_imag_t)
                                env = torch.sqrt(xr.pow(2) + xi.pow(2)).squeeze().cpu().numpy()
                                axs_r[ii + 1].plot(env)
                                axs_r[ii + 1].set_title('filter %d response (envelope)' % fid)
                            except Exception:
                                logging.exception('Failed compute response for filter %d' % fid)
                        fig_r.tight_layout()
                        fig_r.savefig(os.path.join(self.save_dir_sub, f'gabor_val_sample_{si}_responses.jpg'))
                except Exception:
                    logging.exception('Failed to generate validation responses for gabor')

                # 将最优Gabor参数汇总追加至精度输出结果中
                try:
                    param_info = 'Gabor params (best model) - count: %d\n' % sp.shape[0]
                    param_info += 'f0 mean (center frequency / 中心频率): %.6f\n' % (f0.mean())
                    param_info += 'sigma mean (Gaussian width / 高斯宽度): %.6f\n' % (sigma.mean())
                    if theta is not None:
                        # 以角度为单位书写θ均值，方便读取
                        param_info += 'theta mean (degrees / 角度): %.6f\n' % ((theta.mean() * 180.0 / np.pi))
                    #同时保存带有双语键的完整数组
                    try:
                        import json as _json
                        data = {'说明': 'Gabor 参数 / Gabor params', 'f0_中心频率': f0.tolist(), 'sigma_高斯宽度': sigma.tolist()}
                        if theta is not None:
                            data['theta_degrees（度）'] = (theta * 180.0 / np.pi).tolist()
                        with open(os.path.join(self.save_dir_sub, 'gabor_params.json'), 'w', encoding='utf-8') as jf:
                            _json.dump(data, jf, ensure_ascii=False, indent=2)
                    except Exception:
                        logging.exception('Failed to write gabor_params.json')
                    with open(os.path.join(self.save_dir, 'acc output.txt'), 'a', encoding='utf-8') as f:
                        f.write('\n' + param_info)
                except Exception:
                    logging.exception('Failed to append gabor params to acc output')

            # 如果加载了最优模型，则恢复模型状态
            if getattr(self, 'save_best_data', None) is not None and model_backup is not None:
                try:
                    self.model.load_state_dict(model_backup)
                except Exception:
                    logging.exception('Failed to restore model state after gabor analysis')
        except Exception:
            logging.exception('Gabor analysis failed')
