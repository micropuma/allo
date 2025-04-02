import torch
import torch.nn as nn
import torch.nn.functional as F
import allo
import numpy as np

# 定义一个简化的 ResNet 模型（ResNet-18）
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 初始化ResNet-18
resnet_model = ResNet(BasicBlock, [2, 2, 2, 2])

# 设定为评估模式
resnet_model.eval()

# 准备 Allo 相关输入数据
example_inputs = [torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)]

# 使用 Allo 转换 PyTorch 模型为静态计算图
llvm_mod = allo.frontend.from_pytorch(resnet_model, example_inputs=example_inputs, verbose=True)

# 计算预测结果
golden = resnet_model(*example_inputs)

# 将输入转为 numpy 数组
np_inputs = [x.detach().numpy() for x in example_inputs]

# 使用转换后的 LLVM 模型进行推理
res = llvm_mod(*np_inputs)

# 检查输出是否一致
torch.testing.assert_close(res, golden.detach().numpy())
print("Passed!")

# 如果目标是 VHDL 代码，可以生成相应的 HLS 代码
mod = allo.frontend.from_pytorch(resnet_model, example_inputs=example_inputs, target="vhls")
print(mod.hls_code)

