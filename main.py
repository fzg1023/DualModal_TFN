import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import os
import sys
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils import train_utils
import random
import numpy as np 
import torch 
import time

seed = 999
np.random.seed(seed) 
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

mybool = lambda x: x.lower() in ['yes', 'true', 't', 'y', '1']


def optional_float(x):
    """允许命令行用 none/off/null 表示不加噪。"""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {'none', 'null', 'off', 'clean'}:
        return None
    return float(x)


def set_seed(seed: int, deterministic: bool = True):
    """统一设置随机种子；多次独立运行只修改 --seed。"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # 数据集参数
    parser.add_argument('--data_dir', type=str, default=r'D:\Desktop\BJTU_RAO_Bogie_Datasets',
                        help='the directory of the dataset')
    parser.add_argument('--data_name', type=str, default='BJTU-RAO', choices=['UORED', 'BJTU-RAO'],
                        help='the name of the dataset (UORED or BJTU-RAO)')
    parser.add_argument('--data_type', type=str, default='time', choices=['time'],
                        help='the data_type of the dataset')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std'], default='mean-std',
                        help='data normalization methods')
    parser.add_argument('--data_signalsize', type=int, default=2048, help='the name of the data')
    parser.add_argument('--SNR', type=optional_float, default=-6, help='optional SNR in dB; use none/clean to disable. Table 2 should normally use clean data')
    parser.add_argument('--train_SNR', type=optional_float, default=None,
                        help='optional SNR in dB for training inputs; if not set, falls back to --SNR')
    parser.add_argument('--val_SNR', type=optional_float, default=None,
                        help='optional SNR in dB for validation/test inputs; if not set, falls back to --train_SNR')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of dataloader workers')
    parser.add_argument('--test_size', type=float, default=0.3, help='for few-shot analysis')
    parser.add_argument('--seed', type=int, default=999, help='random seed; change only this for repeated runs')
    parser.add_argument('--split_seed', type=int, default=999,help='random seed for train/val/test split')
    parser.add_argument('--single_modality', type=str, default='vib', choices=['vib', 'aco'],
                        help='for single-modal models: vib uses channel 0, aco uses channel 1')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='DualModal_TFN',
                        choices=["Backbone_CNN", "Random_CNN", "TFN_Gabor", "DualModal_TFN"],
                        help='the model to be trained')
    parser.add_argument('--kernel_size', type=int, default=11, help='the kernel size of traditional conv layer')
    parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoint',
                        help='the directory to save the models and the results')

    # 双模态参数
    parser.add_argument('--use_decision_fusion', type=mybool, default="True",
                        help='DualModal_TFN: use decision-level fusion in addition to feature fusion')
    parser.add_argument('--decision_alpha', type=float, default=0.1,
                        help='DualModal_TFN: weight for auxiliary branch logits')
    parser.add_argument('--aux_loss', type=mybool, default="True",
                        help='DualModal_TFN: add auxiliary cross-entropy losses for each branch')
    parser.add_argument('--aux_weight', type=float, default=0.2,
                        help='weight of each unimodal auxiliary CE loss')

    # 功能模型参数
    parser.add_argument('--mid_channel', type=int, default=64, help='the channel number of preprocessing layer')
    parser.add_argument( '--gabor1d_kernel_size',type=int,default=63,
                        help='kernel size of 1D Gabor filters')
    parser.add_argument('--gabor2d_kernel_size', type=int, default=15,
                        help='2D Gabor kernel size; decoupled from mid_channel to avoid oversized 127x127 kernels')
    parser.add_argument('--dual_gabor_mode', type=str, choices=['auto', '1d', '2d'], default='auto',
                        help='DualModal_TFN Gabor front-end: auto uses fast 1d for noisy runs and 2d for clean runs')
    parser.add_argument('--dual_balanced_snr', type=mybool, default='True',
                        help='DualModal_TFN only: apply requested SNR after normalization, independently per modality, to avoid channel-scale distortion')
    # v5: internal learnable two-modal spectral branch. No SVM/teacher is used.
    parser.add_argument('--dual_internal_spectral', type=mybool, default='True',
                        help='DualModal_TFN noisy runs: enable the internal learnable two-modal spectral branch')
    parser.add_argument('--internal_spectral_bins', type=int, default=128,
                        help='pooled FFT bins per modality for the internal spectral branch')
    parser.add_argument('--internal_spectral_hidden', type=int, default=128,
                        help='hidden width of the lightweight internal spectral branch')
    parser.add_argument('--internal_spectral_alpha', type=float, default=0.80,
                        help='initial convex-fusion weight assigned to the internal spectral branch')
    parser.add_argument('--internal_spectral_aux_weight', type=float, default=0.60,
                        help='auxiliary CE weight for the internal spectral classifier')
    parser.add_argument('--internal_spectral_pretrain_epochs', type=int, default=15,
                        help='one-time train-split-only pretraining epochs for the internal spectral head')
    parser.add_argument('--internal_spectral_pretrain_repeats', type=int, default=6,
                        help='target-SNR noisy views per train sample used for spectral-head pretraining')
    parser.add_argument('--internal_spectral_pretrain_lr', type=float, default=1e-3,
                        help='learning rate for one-time spectral-head pretraining')

    # v4/v4.1 teacher switches are retained only for command-line compatibility.
    # v5 intentionally disables external teacher/KD so final inference is one DualModal_TFN model.
    parser.add_argument('--dual_spectral_teacher', type=mybool, default='False', help=argparse.SUPPRESS)
    parser.add_argument('--teacher_bins', type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument('--teacher_fit_repeats', type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument('--teacher_target_repeats', type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument('--teacher_C', type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument('--distill_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--distill_temperature', type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument('--distill_conf_floor', type=float, default=0.70, help=argparse.SUPPRESS)
    parser.add_argument('--teacher_logit_scale', type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument('--distill_start_epoch', type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument('--distill_ramp_epochs', type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument('--ema_start_epoch', type=int, default=5,
                        help='start EMA after the internal spectral branch and TFN have begun joint training')
    parser.add_argument('--dual_ema', type=mybool, default='True',
                        help='DualModal_TFN noisy runs: compare live and EMA pure-model validation weights')
    parser.add_argument('--ema_decay', type=float, default=0.990,
                        help='maximum EMA decay')

    # Older switches remain accepted but are ignored by v5.
    parser.add_argument('--dual_spectral_residual', type=mybool, default='False', help=argparse.SUPPRESS)
    parser.add_argument('--spectral_alpha', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--spectral_aux_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--dual_vib_prior', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dual_vib_floor', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--dual_spectral_expert', type=mybool, default='False', help=argparse.SUPPRESS)
    parser.add_argument('--spectral_expert_bins', type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument('--spectral_expert_repeats', type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument('--spectral_expert_C', type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument('--spectral_expert_nn_residual', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--spectral_expert_neural_aux', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--dual_gabor2d_lr', type=float, default=5e-4,
                        help='DualModal 2D Gabor learning rate; 2D theta/sigma must not be frozen at the 1D 1e-5 rate')
    parser.add_argument('--clamp_flag', type=mybool, default="True", help='flag to limit the superparams of TFconv layer')

    # 优化参数
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam', 'RMSprop'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--gabor_lr', type=float, default=1e-5, help='learning rate for Gabor super-parameters')
    parser.add_argument('--gabor_warmup_epochs',type=int,default=0,help='epochs to freeze Gabor parameters before fine-tuning')
    parser.add_argument('--gabor_reg', type=float,default=1e-3,help='regularization strength keeping Gabor params near initialization')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='learning rate scheduler parameter for step and exp')  # 0.99
    parser.add_argument('--steps', type=str, default='5', help='the learning rate decay interval for step and stepLR; v5 default=5 for low-SNR stability')

    # save, load and display information
    parser.add_argument('--save_model', type=mybool, default="False", help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=100 , help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=5, help='the interval of log training information')
    args = parser.parse_args()
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return args


if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print("current dir: %s" % os.path.curdir)

    args = parse_args()
    set_seed(args.seed, deterministic=True)

# 处理 SNR 参数优先级：优先使用 --train_SNR/--val_SNR，若未提供则回退到 --SNR
    if args.SNR is not None and (args.SNR <= -1e2 or args.SNR >= 1e2):
        args.SNR = None
    # 若未单独指定 train_SNR，则使用通用 SNR
    if args.train_SNR is None:
        args.train_SNR = args.SNR
    # 若未单独指定 val_SNR，则设置与 train_SNR 相同（使验证与训练使用相同 SNR）
    if args.val_SNR is None:
        args.val_SNR = args.train_SNR

    # 对 train_SNR/val_SNR 做范围校验（与之前对 args.SNR 一致的阈值）
    if args.train_SNR is not None and (args.train_SNR <= -1e2 or args.train_SNR >= 1e2):
        args.train_SNR = None
    if args.val_SNR is not None and (args.val_SNR <= -1e2 or args.val_SNR >= 1e2):
        args.val_SNR = None


    # 单次运行：使用 args 中的 SNR/train_SNR/val_SNR 设置
    args_run = args

    # 确保信噪比字段数值处于预期范围内（超出范围则视为无数据）
    if getattr(args_run, 'SNR', None) is not None and (args_run.SNR <= -1e2 or args_run.SNR >= 1e2):
        args_run.SNR = None
    if getattr(args_run, 'train_SNR', None) is None:
        args_run.train_SNR = args_run.SNR
    if getattr(args_run, 'val_SNR', None) is None:
        args_run.val_SNR = args_run.train_SNR
    if getattr(args_run, 'train_SNR', None) is not None and (args_run.train_SNR <= -1e2 or args_run.train_SNR >= 1e2):
        args_run.train_SNR = None
    if getattr(args_run, 'val_SNR', None) is not None and (args_run.val_SNR <= -1e2 or args_run.val_SNR >= 1e2):
        args_run.val_SNR = None
    #创建实验保存目录
    sub_dir = args_run.model_name + '-' + args_run.data_name + '-' + args_run.data_type + '-trainSNR{}'.format(
        'None' if args_run.train_SNR is None else str(args_run.train_SNR)) + '-' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args_run.checkpoint_dir, sub_dir).replace('\\', '/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置日志记录器
    setlogger(os.path.join(save_dir, 'training.log'))

    # 对外开放保存目录，供外部脚本与日志使用
    print(f"SAVE_DIR:{save_dir}")
    sys.stdout.flush()

    # 记录本次运行的参数
    for k, v in args_run.__dict__.items():
        logging.info("<args> {}: {}".format(k, v))

    # 初始化训练器并完成前置设置
    trainer = train_utils(args_run, save_dir)
    trainer.setup()

    # 训练/评估模型
    if args_run.max_epoch <= 0:
        logging.info('max_epoch <= 0, running setup-only evaluation (no training).')
        trainer.train()

    else:
        time_start_train = time.time()
        trainer.train()
        logging.info("<training time>: {:.3f}".format(time.time() - time_start_train))

    # 绘制结果（将包含双语标签）
    trainer.plot_save()
