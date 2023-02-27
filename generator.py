import torch
import torch.nn as nn
class ConvBlock(nn.Module):

    def __init__(self,input,output,down=True,use_act=True,**kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input,out_channels=output,padding_mode='zeros',**kwargs)
            if down
            else nn.ConvTranspose2d(in_channels=input,out_channels=output,**kwargs),
            nn.InstanceNorm2d(output),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self,x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            ConvBlock(input=channels,output=channels,kernal_size = 3,padding = 1),
            ConvBlock(input=channels,output=channels,use_act=False,kernal_size = 3,padding = 1)
        )

    def forward(self,x):
        return x+self.block(x)

class Generator(nn.Module):
    def __init__(self,img_channel,num_feature=64,num_residual=9):
        super(Generator,self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=img_channel,out_channels=num_feature,kernel_size=7,stride=1,padding=3,padding_mode='zeros'),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(input=num_feature,output=num_feature*2,kernal_size=3,stride=2,padding=1),
                ConvBlock(input=num_feature*2, output=num_feature * 4, kernal_size=3, stride=2,padding=1),
            ]
        )

        self.residual_block = nn.Sequential(
            *[ResidualBlock(num_residual*4) for _ in range(num_residual)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(input=num_feature * 4, output=num_feature * 2, kernal_size=3, stride=2, padding=1, ouput_padding=1),
                ConvBlock(input=num_feature * 2, output=num_feature * 1, kernal_size=3, stride=2, padding=1, ouput_padding=1),
            ]
        )

        self.last = nn.Conv2d(in_channels=num_feature,out_channels=img_channel,kernel_size=7,stride=1,padding=3,padding_mode='zeros')

    def forward(self,x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_block(x)
        for layer in self.up_blocks:
            x = layer(x)

        return torch.tanh(self.last(x))

