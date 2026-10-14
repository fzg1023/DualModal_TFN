import torch
nn = torch.nn
from utils.mysummary import summary

class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10,kernel_size=15,**kwargs):#可变参数，以便后续扩展
        super(CNN, self).__init__()
        #定义第一层网络
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=kernel_size, bias=True), # 通道数16, 1010（序列长度=输入长度1024-核大小15+1)
            nn.BatchNorm1d(16),#批量归一化 对16个通道的特征做归一化，加速训练、缓解梯度消失
            nn.ReLU(inplace=True))#激活函数ReLU，增加网络非线性表达能力 inplace=True表示直接在输入上进行操作，节省内存

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, bias=True),  # 输入（上层）16、输出通道32, 输出长度=1010-3+1=1008
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 一维最大池化层 核大小2、步长2 输出通道32, 输出长度1008/2=504，通道数保持32

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, bias=True),  # 输入（上层）32、输出通道64, 输出长度=504-3+1=502
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, bias=True),  # 输入（上层）64、输出通道128, 输出长度=502-3+1=500
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))  # 一维自适应最大池化，无论输入长度多少，输出长度固定为4，通道数保持128（避免输入长度变化导致全连接层输入维度不匹配）
        #全连接层（线性层）
        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),#第一层：输入维度=128通道×4长度=512，输出维度256
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),#第二层：输入256，输出64
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, out_channels)#最终分类层，输入64，输出类别数out_channels（默认为10）

    def forward(self, x):
        if len(x.shape) == 4:#处理输入维度兼容问题：若输入为4维张量（批次大小、通道数、序列长度、1 [batch,1,1,1024]），则去掉最后一个维度
            x = torch.squeeze(x)
       #依次将输入传入layer1~layer4，执行卷积+归一化+激活+池化的前向计算，进行特征提取，最后通过全连接层输出分类结果
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)#展平张量，将卷积层输出的特征图转换为全连接层输入的格式，保持批次大小不变，其他维度展平为一维向量
        x = self.layer5(x)
        x = self.fc(x)#最终分类层
        return x


if __name__ == '__main__':
    model = CNN()
    info = summary(model, (1, 1024), batch_size=-1, device="cpu")#-1不指定批次大小
    print(info)