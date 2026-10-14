"""
该脚本用于从日志文件中提取训练信息并保存至Excel，同时绘制精度曲线图。
"""
import os
from process_utils.processlib import ExtractInfo
from process_utils.processlib import acc2csv
from process_utils.PlotAccuracy import main as PlotaccMain

def main(root = '../checkpoint/Acc-UORED',logname = "training.log"):
    # 从日志文件中提取训练信息
    subdirs = next(os.walk(root))[1]
    Info = []
    for subdir in subdirs:
        if subdir != "postfiles":
            filepath = os.path.join(root, subdir, logname)
            Params, Dict = ExtractInfo(filepath)
            Info.append(Params)
    #  保存至Excel并绘图
    save_dir = os.path.join(root, 'postfiles')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    acc2csv(os.path.join(save_dir, "1-Acc_statistic.xlsx"), Info,focus_column = ['model_name','mid_channel'])
    PlotaccMain(os.path.join(save_dir, "1-Acc_statistic.xlsx"),focus_column = ['model_name','mid_channel'])

if __name__ == '__main__':
    # 设置当前目录
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print("current dir: %s" % os.path.curdir)
    # 准确率统计
    main(root = '../checkpoint/Acc-UORED')