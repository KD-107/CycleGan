import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,input,output,stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input,
                out_channels=output,
                stride=stride,
                kernel_size=4,
                padding=1,
                bias=True,
                padding_mode='zeros'
            ),
            nn.InstanceNorm2d(num_features=output),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)


class Discriminator(nn.Module):

    def __init__(self,input,features = [64,128,256,512]):
        super(Discriminator,self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=input,
                out_channels=features[0],
                stride=2,
                padding=1,
                kernel_size=4,
                padding_mode='zeros'
            ),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(input=in_channels,output=feature,stride=1 if feature==features[-1] else 2))
            in_channels = feature

        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=4,stride=1,padding=1,padding_mode='zeros'))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))