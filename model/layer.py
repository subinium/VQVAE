import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dInit(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=(1,1),
                padding=None,
                dilation=(1,1),
                bias=True,
                w_init_gain='linear',
                param=0.02):
        super().__init__()
        if padding is None:
            assert(kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1)
            padding = (int(dilation[0] * (kernel_size[0] - 1) / 2), int(dilation[1] * (kernel_size[1] - 1) / 2))

        self.conv = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)

        nn.init.xavier_uniform_(self.conv.weight, 
                                gain=nn.init.calculate_gain(w_init_gain, param))

    def forward(self, input):
        return self.conv(input)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(True),
            Conv2dInit(in_channels, num_residual_hiddens, 3, 1, 1),
            nn.ReLU(True),
            Conv2dInit(num_residual_hiddens, num_hiddens, 3, 1, 1),
        )

    def forward(self, input):
        return input + self.model(input) 

class ResidualBlocks(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        model = [ResidualBlock(in_channels, num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return F.relu(self.model(input))