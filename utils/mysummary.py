"""
分析PyTorch神经网络模型的结构与资源消耗
遍历模型所有层，收集每层的类型、输入 / 输出形状、参数数量（可训练 / 不可训练）；
统计模型总参数、可训练参数、非可训练参数；
估算模型的输入 / 输出 / 参数的显存占用（MB），以及前向 / 反向传播的显存消耗；
同时将信息打印到控制台，并返回拼接好的信息字符串（便于日志保存 / 二次处理）。
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda"):
    return_info = ""

    def register_hook(module):
        def hook(module, input, output):
            #1.获取当前层的类名
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            #2.生成层的唯一标识
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            #3.初始化当前层的信息字典
            summary[m_key] = OrderedDict()
            #4.记录输入形状：修正batch-size（替换维度0为指定batch-size）
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            #5.记录输出形状：处理多输出和单输出情况
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            #6.统计当前层的参数数量（权重+偏置）
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad #标记是否可训练
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params #保存当前层参数总数
        #7.过滤不需要注册钩子的模块（避免重复/冗余)
        if (
            not isinstance(module, nn.Sequential)# 排除Sequential容器
            and not isinstance(module, nn.ModuleList)# 排除ModuleList容器
            and not (module == model)# 排除模型根节点
        ):
            hooks.append(module.register_forward_hook(hook))# 注册钩子并保存句柄
    #设备合法性检查与张量类型选择
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor# CUDA可用则用CUDA张量
    else:
        dtype = torch.FloatTensor# 否则用CPU张量

    # 生成测试输入张量
    #处理多输入：如果input_size是元组，转为列表（适配多输入模型）
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # 生成批量大小为2的随机测试输入（batch_size=2适配BatchNorm等层的正常运行）
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # 初始化：有序字典存储每层信息，列表存储钩子句柄
    summary = OrderedDict()
    hooks = []

    # 遍历模型所有子模块，注册钩子（触发register_hook）
    model.apply(register_hook)

    #前向传播：触发所有钩子的hook函数，收集层信息
    model(*x)

    # 移除钩子：避免钩子残留影响模型后续运行
    for h in hooks:
        h.remove()
    #打印/拼接层信息表头
    print("----------------------------------------------------------------")
    return_info += "----------------------------------------------------------------\n"

    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    return_info += line_new +"\n"
    print("================================================================")
    return_info += "----------------------------------------------------------------\n"
    #遍历统计每层信息
    total_params = 0 #总参数数
    total_output = 0 #总输出元素数（用于显存计算）
    trainable_params = 0 #可训练参数数
    for layer in summary:
        #格式化打印每层信息
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        #累计统计
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])#计算输出元素总数
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        #打印+拼接
        print(line_new)
        return_info += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return_info += "================================================================\n"
    return_info += "Total params: {0:,}\n".format(total_params)
    return_info += "Trainable params: {0:,}\n".format(trainable_params)
    return_info += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    return_info += "----------------------------------------------------------------\n"
    return_info += "Input size (MB): %0.2f\n" % total_input_size
    return_info += "Forward/backward pass size (MB): %0.2f\n" % total_output_size
    return_info += "Params size (MB): %0.2f\n" % total_params_size
    return_info += "Estimated Total Size (MB): %0.2f\n" % total_size
    return_info += "----------------------------------------------------------------\n"
    return_info += "================================================================\n"
    return return_info
