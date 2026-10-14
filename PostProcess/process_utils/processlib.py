"""
自动化从模型训练日志文件中提取关键训练指标
（如学习率、训练 / 验证损失、准确率、训练时长等）
将提取的多组实验信息进行结构化整理、统计分析（均值、标准差）后，保存为 Excel 文件
最终实现实验结果的高效分析与对比
"""
import os
import numpy as np
from datetime import datetime
import pandas as pd


def ExtractInfo(filepath,append_acc = True):
    """
    从日志文件中提取训练信息
    """
    start_time, record_time = None, None
    print(os.path.abspath(filepath))# 打印日志文件绝对路径，便于调试/确认文件位置

    # 存储逐轮（Epoch）的训练/验证指标（列表形式）
    Dict = {'current_lr': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    # 存储汇总类参数，先初始化filename为日志所在目录的最后一级名称
    Params = {'filename': os.path.split(os.path.split(filepath)[0])[-1]}
    with open(filepath, 'r',encoding='utf8') as f:# 打开文件（utf8编码避免乱码）
        temp = f.readline()
        flag_temp = True
        # 第一步：提取Epoch前的实验参数（如模型名、学习率等）
        while '-----Epoch' not in temp:  # # 循环直到读取到Epoch行
            if ": " in temp[15:] and flag_temp:# 匹配"键: 值"格式的参数行
                key = temp.split(": ")[0].split(" ")[-1]# 提取参数名
                value = temp.split(": ")[1].replace("\n", '')# 提取参数值
                Params[key] = value
            if '---------------------------------' in temp:# 分隔线后停止提取参数
                flag_temp = False
            temp = f.readline()
        # 第二步：提取每轮Epoch的训练/验证指标
        while temp != '':  # 循环到文件末尾
            if '-----Epoch' in temp:  # 匹配Epoch行，开始提取本轮指标
               # 提取当前学习率（转浮点数
                Dict['current_lr'].append(float(f.readline().split('current lr: ')[1].split(':')[-1].strip('\n[]')))
                temp = f.readline()
                # 初始化训练开始时间（仅第一轮Epoch执行）
                if not start_time:
                    start_time = datetime.strptime(temp[:14], '%m-%d %H:%M:%S')
                # 跳过临时信息行
                while "<tempinfo>" in temp:  # skip tempinfo
                    temp = f.readline()
                # 提取训练损失、训练准确率
                Dict['train_loss'].append(float(temp.split('train-Loss: ')[1].split(' ')[0]))
                Dict['train_acc'].append(float(temp.split('train-Acc: ')[1].split(' ')[0].replace(',', '')))
                temp = f.readline()
                # 提取验证损失、验证准确率
                Dict['val_loss'].append(float(temp.split('val-Loss: ')[1].split(' ')[0]))
                Dict['val_acc'].append(float(temp.split('val-Acc: ')[1].split(' ')[0].replace(',', '')))
                # 记录本轮Epoch结束时间
                end_time = datetime.strptime(temp[:14], '%m-%d %H:%M:%S')
           # 提取日志中直接记录的训练总时间
            if '<training time>: ' in temp:  # extract training time
                record_time = float(temp.split('<training time>: ')[1].split(' ')[0].replace(',', ''))
            temp = f.readline()
    # 将Dict中的列表转为numpy数组（便于数值计算）
    for key in Dict.keys():  # 将列表转换为numpy数组
        if key != 'params':# 冗余判断（Dict无该键），兼容扩展
            Dict[key] = np.array(Dict[key])
    # 计算训练总时长（优先用日志记录的时间，否则用时间差
    Params['train_time'] = record_time if record_time else (end_time - start_time).total_seconds()
    # 追加最大/最终验证准确率（若append_acc=True）
    if append_acc:
        Params['max acc'] = max(Dict['val_acc'])# 最优验证准确率
        # 计算最终准确率的取数长度（不超过5、不低于1、不超过总轮数的50%）
        final_len = int(max(min(len(Dict['val_acc'])*0.5,5),1))
        Params['final acc'] = Dict['val_acc'][-final_len:].mean()# 最后N轮验证准确率均值
    return Params,Dict # 返回汇总参数、逐轮指标


def acc2csv(savepath,data,focus_column = ['model_name']):
    """
    将训练信息保存至Excel
    """
    # 步骤1：原始数据转为DataFrame，排序并转换数值类型
    df1 = (pd.DataFrame(data).sort_values(by=focus_column).reset_index(drop=True))
    df1 = df1.apply(pd.to_numeric,errors='ignore')# 数值列转数值类型，非数值列保留

    # 步骤2：生成df2（全列分组的均值+标准差，扁平列名
    number = 3 if 'train_time' in df1.columns else 2
    col = df1.drop(['filename','checkpoint_dir'],axis=1).columns.to_list()[:-number]
    df2 = df1.drop(['filename','checkpoint_dir'],axis=1).set_index(col)
    df2 = pd.concat([df2.groupby(col).mean(),df2.groupby(col).std()],axis=1,keys=['mean','std']).reset_index()
    df2.columns = [f'{i} {j}'.strip() for i, j in df2.columns]
    # 步骤3：生成df3（聚焦列分组的均值+标准差，双层列名）
    try:
        temp = df1.loc[:,focus_column+['train_time','max acc', 'final acc']].set_index(focus_column)
    except:# 兼容无train_time列的情况
        temp = df1.loc[:, focus_column + ['max acc', 'final acc']].set_index(focus_column)
    meanstddf = pd.concat([temp.groupby(focus_column).mean(),temp.groupby(focus_column).std()],axis=1,keys=['mean','std']).reset_index()
    df3 = meanstddf.set_index(focus_column)
    df3.columns.names = ['mean/std','max/final']# 列名分层，提升可读性

    # 步骤4：生成df4（聚焦列分组的均值+标准差，扁平列名）
    df4 = meanstddf
    df4.columns = [f'{i} {j}'.strip() for i, j in df4.columns]

    df_save = {'df1': df1, 'df2': df2, 'df3': df3, 'df4': df4}

    # 步骤5：保存4个DataFrame到Excel的不同sheet
    with pd.ExcelWriter(savepath) as writer:
        for item in ['df1', 'df2', 'df3', 'df4']:
            df_save[item].to_excel(writer, item)


if __name__ == '__main__':
    ExtractInfo(r'D:\Desktop\DualModal_TFN-main - 10.1AA\checkpoint\DualModal_TFN-UORED-time-trainSNR-6-0612-144232\training.log', append_acc=True)
