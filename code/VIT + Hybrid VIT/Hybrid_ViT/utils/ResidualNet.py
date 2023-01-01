import torch
from torch import nn



class ResidualBlock(nn.Module):   # 对应18层和34层的残差结构
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None, re_zero=False):  # out_channels也就是卷积核的个数
        super(ResidualBlock, self).__init__()
        
    
        self.layers = nn.Sequential(    # OUTPUT = INPUT
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding=1, bias=False),
                    # stride = 1时对应的是实线结构，是不需要改变的，如果stride = 2就对应虚线的那个，
                    # stride = 1时，output = (input - 3 + 2 * 1)/1 + 1 = input，前后的shape是不变的
                    # stride = 2时，output = (input - 3 + 2 * 1)/2 + 1 = input/2 + 0.5 = input/2 (向下取整)


            nn.BatchNorm2d(out_channels),
            # 参数为 输入特征矩阵的深度，也就是out_channel

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # 无论是实线还是虚线，第二层卷积层的stride都固定为1

            nn.BatchNorm2d(out_channels)
        )  
        # out_size = in_size

        self.residual = shortcut
        self.re_zero = re_zero
        if re_zero:
            self.alpha = nn.Parameter(torch.zeros(1))
        self.activ = nn.GELU()


    def forward(self, x):
        left = self.layers(x)

        right = self.residual(x) if self.residual else x
        # shortcut = nn.Conv2d(inchannel, outchannel, 1, stride, bias=False) + nn.BatchNorm2d(outchannel) 
        if self.re_zero:
            right = right * self.alpha


        out = left + right
        return self.activ(out)





class Res34(nn.Module):
    def __init__(self, args, in_channels, out_channels=None):
        super(Res34, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True),     # 112 x 112 x 64
            # in_channels是固定为64的，然后卷积核是7，out_size = in_size / 2
            # 对应最开始的7 x 7，64通道，stride = 2，padding设置为3，让最初的宽高变为原来的一半


            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d(3, 2, 1)  # out_size = in_size//2 向上取整    56 x 56 x 64
            # 对应第二层的maxpool
        )

        self.re_zero = args.re_zero
        self.layer1 = self._make_layer(64, 128, 3)    # 56 x 56 x128
        self.layer2 = self._make_layer(128, 256, 4, stride=2)   # 28 x 28 x 256
        self.layer3 = self._make_layer(256, out_channels, 6, stride=2)   # 14 x 14 x outchannel
        

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):  
        # in_channel对应第一层的卷积核的个数，50层以上的就是那个最少的channel数
        # block_num：包含多个残差结构，一个list
        
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = [ResidualBlock(inchannel, outchannel, stride, shortcut, self.re_zero)]  
        # 引入第一层的ResidualBlock

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, re_zero=self.re_zero))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.softmax(x)
        return x












class CNN(nn.Module):
    def __init__(self, args, inchannel, outchannel):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=213, stride=1, padding=1, bias=True), 
            
        )

    def forward(self, x):
        x = self.layers(x)
        return x
