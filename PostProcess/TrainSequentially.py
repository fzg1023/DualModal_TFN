"""
该脚本用于按顺序训练模型。
运行此脚本前，请先在<main.py>中将<--data_dir>设置为UORED数据集所在的目录。
"""
import os, sys

if __name__ == '__main__':
    # 设置当前目录
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print("current dir: %s" % os.path.curdir)
    # 准备命令行
    model_list = ["Backbone_CNN", "Random_CNN", "TFN_Gabor", "DualModal_TFN"]
    max_epoch = 50
    mid_channel = [16, 32, 64, 128]
    command_lines = []
    SNR = 1e3
    checkpoint_dir = os.path.abspath(r'..\checkpoint\Acc-UORED').replace('\\', '/')
    data_name = 'UORED'

    for item in model_list:
        for item2 in mid_channel:
            if item == "Backbone_CNN" and item2 != 16:
                continue
            line = f' --data_name {data_name:s} --SNR {SNR:.2f}' + \
                   f' --model_name {item:s} --mid_channel {str(item2):s}' \
                   f' --checkpoint_dir "{checkpoint_dir:s}" --save_model False  --max_epoch {max_epoch:d}'
            command_lines.append(line)

    """
    轮次：每个模型的运行次数，为节省时间我们将其设为2。
    起始序号：命令行的起始索引（默认值：0）
    """
    round = 2;
    start = 0
    for i, item in enumerate(command_lines):
        for p in range(round):
            if i * round + p + 1 <= start:
                continue
            print('-----' * 10)
            print('process of total: {:^3d}/{:^3d} \ncommand line: {:s}'.format(i * round + p + 1,
                                                                                len(command_lines) * round, item))
            temp = '%s "%s/main.py" %s' % (sys.executable, os.path.abspath('../'), item)
            print("Commandline: %s" % temp)
            # 在终端中运行命令行（使用main.py训练模型）
            os.system(temp)
            print('-----' * 10)
