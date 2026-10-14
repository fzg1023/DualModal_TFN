"""
将结构化的模型测试准确率数据（含均值和标准差）可视化成带误差棒的柱状图
重点对比不同模型、不同预处理层通道数下的测试准确率表现
支持加入基准模型（Backbone）做对照
最终生成符合学术绘图规范（固定字体、高清分辨率、统一样式）的准确率对比图
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def setseaborn():
    # set color
    current_cmap = sns.color_palette("deep")#deep调色板
    sns.set(style="whitegrid")#白色网格
    # 细化样式：ticks风格、上下文为notebook、字体为Times New Roman（学术常用）、调色板、字体缩放
    sns.set(style="ticks", context="notebook", font='Times New Roman', palette=current_cmap, font_scale=2)
    return current_cmap# 返回调色板，供后续绘图使用

#配置 matplotlib 全局默认参数
def setdefault():
    mpl.rcParams['axes.linewidth'] = 0.4#坐标轴线条宽度
    mpl.rcParams['font.family'] = 'Times New Roman'# 全局字体（覆盖seaborn，确保统一）
    mpl.rcParams['xtick.labelsize'] = 6# x轴刻度标签字体大小
    mpl.rcParams['ytick.labelsize'] = 6# y轴刻度标签字体大小
    mpl.rcParams['figure.figsize'] = [8 / 2.54, 6 / 2.54]# 图尺寸（8cm × 6cm，÷2.54是厘米转英寸）
    mpl.rcParams['figure.dpi'] = 1000# 图片分辨率（1000dpi，高清，适合论文发表）


def setaxesLW(myaxes, axes_lw=1, tick_len=3, tick_lw=None):
    if not tick_lw:
        tick_lw = axes_lw / 3 * 2 # 若未指定刻度线宽，按坐标轴宽度比例计算
    # 遍历四个方向的坐标轴（上/左/下/右），设置线条宽度
    for item in ['top', 'left', 'bottom', 'right']:
        myaxes.spines[item].set_linewidth(axes_lw)
    # 设置刻度的线宽和长度
    myaxes.tick_params(width=tick_lw, length=tick_len)


def PlotAcc(Datas, labels, ticklabels, savename, BackboneData=None, width=0.2, ylim=[75, 100],
            title='The channel number of preprocessing layer', xlabel='Model', ylabel='Test accuracy (%)', ygap=5,
            elinewidth=0.7, ncol=5):
    #样式初始化
    cmap = np.array(setseaborn()) # 获取seaborn调色板并转为数组
    setdefault() # 加载matplotlib默认样式
    color_alpha = 1 # 颜色透明度（1=不透明）
    cmap = cmap * color_alpha + np.array([1.0, 1, 1]) * (1 - color_alpha) # 调整透明度（此处无实际变化）
    labelsize, ticksize, legend_size = 6, 6, 6 # 定义标签/刻度/图例字体大小
    err_kw = {'elinewidth': elinewidth, 'ecolor': 'k'} # 误差棒参数（线宽、颜色为黑色）
    #初始化绘图对象
    X = np.arange(1, Datas.shape[0] + 1) # 生成x轴基础位置（1到模型数
    fig = plt.figure() # 创建图对象
    ax = fig.add_subplot() # 创建轴对象（子图）
    Bars = [] # 存储柱状图对象，备用
    # 绘制核心数据（不同模型+不同参数的柱状图）
    for i in range(Datas.shape[1]):# 遍历参数配置（如通道数）
        values = Datas[:, i, 0]
        errors = Datas[:, i, 1] if Datas.ndim == 3 and Datas.shape[2] > 1 else None
        # 计算每个柱的x位置（避免重叠），绘制柱状图：x位置、均值、宽度、标签、颜色、误差棒（若存在）
        temp = ax.bar(X + width * (i + 1 / 2 - Datas.shape[1] / 2), values, width, label=labels[i],
                      color=cmap[i + 1], yerr=errors, error_kw=err_kw)
        Bars.append(temp)# 保存柱状图对象
    #处理图例（重新排序，避免过长）
    h, l = plt.gca().get_legend_handles_labels()# 获取图例的手柄和标签
    r, c = int(np.ceil(len(l) / ncol)), ncol# 计算图例的行/列数
    order = [c * (i % r) + (i // r) for i in range(len(l))]# 重新排序图例（按行列重排，优化布局）
    # 设置图例：重排后的手柄/标签、字体大小、位置（图上方居中）、列数、标题
    leg = ax.legend([h[i] for i in order], [l[i] for i in order], prop={'size': legend_size},
                    bbox_to_anchor=(0.5, 1.05), loc='lower center', borderaxespad=0, ncol=ncol)
    leg.set_title(title, prop={'size': legend_size})# 设置图例标题

    # 绘制基准模型（BackboneData）
    if BackboneData is None:
        #无基准模型：调整x轴范围的起始/结束值
        start = 1 - width * Datas.shape[1] / 2 - 0.1
        end = len(ticklabels) + width * Datas.shape[1] / 2 + 0.1
    else:
        # 有基准模型：绘制基准模型的柱状图（x=0位置）
        backbone_values = np.asarray(BackboneData, dtype=float).reshape(-1)
        bar_height = backbone_values[0]
        bar_error = backbone_values[1] if len(backbone_values) > 1 else None
        temp = ax.bar([0], bar_height, width, color=cmap[0], yerr=bar_error, error_kw=err_kw)
        Bars.append(temp)
        # 调整x轴范围（包含基准模型）
        start = 0 - width / 2 * 1 - 0.1
        end = len(ticklabels) - 1 + width * Datas.shape[1] / 2 + 0.1
    # 坐标轴配置
    ax.set_xlabel(xlabel, fontsize=labelsize)# x轴标签
    ax.set_ylabel(ylabel, fontsize=labelsize)# y轴标签
    # 设置x轴刻度位置（有基准模型则刻度从0开始，无则从1开始）
    ax.set_xticks(np.arange(len(ticklabels)) if not BackboneData is None else np.arange(len(ticklabels)) + 1)
    ax.set_xticklabels(ticklabels, fontsize=ticksize)# x轴刻度标签
    ax.set_xlim([start, end])# x轴范围
    if ylim:
        ax.set_ylim(ylim)# y轴范围
    if ygap:
        # 设置y轴刻度（按间隔ygap生成），并格式化标签为整数
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, ygap))
        ax.set_yticklabels(['{:d}'.format(int(item)) for item in np.arange(ylim[0], ylim[1] + 0.01, ygap)],
                           fontsize=ticksize)
    else:
        ax.tick_params(axis='y', labelsize=8)# 无间隔时调整y轴刻度字体
    ax.grid(axis='y', linewidth=0.5)# 仅显示y轴网格，增强可读性
    setaxesLW(ax, 0.8)# 调用函数设置坐标轴/刻度线宽
     # 保存图片
    fig.tight_layout(pad=0.2)# 自动调整布局，避免标签重叠
    fig.savefig(savename.replace('.jpg', '-withlegend.jpg'))# 保存图片（替换后缀，标记含图例）

#读取 Excel 数据、预处理数据、调用PlotAcc()绘图
def main(filedir, focus_column=['model_name', 'mid_channel']):
    mid_channel = [16, 32, 64, 128]# 关注的中间通道数
    # 读取Excel数据（df4工作表
    temp = pd.read_excel(filedir, sheet_name='df4')
    # 筛选列：仅保留关注列 + 准确率均值
    data = temp.loc[:, focus_column + ['mean max acc(%)']]
    data = data.set_index(focus_column)# 设置索引为模型名+通道数，方便筛选
    # 定义模型相关名称（用于x轴标签）
    model_names = ["Random_CNN", "TFN_Gabor", "DualModal_TFN"]# 原始模型名
    model_new_names = ["Backbone\n_CNN", "Random\n_CNN", "TFN\n_Gabor", "DualModal\n_TFN"]# 显示名（换行优化布局）
    backbone_name = "Backbone_CNN"# 基准模型名
    #提取基准模型数据（计算均值，降维）
    CNN_Data = data.loc[backbone_name, :].to_numpy().mean()
    #提取核心模型数据（按模型名+通道数筛选）
    Datas = []
    for item in model_names:
        model_values = data.loc[item, :].loc[mid_channel].to_numpy().reshape(-1, 1)
        Datas.append(model_values)# 提取每个模型不同通道数的准确率
    Datas = np.array(Datas)# 转为三维数组
    # 调用核心绘图函数
    PlotAcc(Datas, ['16', '32', '64', '128'], model_new_names, os.path.join(os.path.split(filedir)[0], '2-TestAcc.jpg'),
            CNN_Data)
if __name__ == "__main__":
    # Excel和脚本在同一文件夹，直接写相对路径
    filedir = r"D:\Desktop\DualModal_TFN-main - 10.1AAAA\PostProcess\process_utils\accuracy_data.xlsx"
    # Windows绝对路径写法示例（用r前缀避免转义）
    # filedir = r"C:\Users\你的用户名\Desktop\accuracy_data.xlsx"
    main(filedir)
