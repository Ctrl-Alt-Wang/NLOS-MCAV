import torch
from torch import nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if not mid_channels:
            self.mid_channels=out_channels

        self.conv1 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,padding=1,bias=False)
        self.double_conv=nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=5,padding=1,bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
         )
    def forward(self,x):
        x = self.conv1(x)
        return x
class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Down, self).__init__()
        self.inchannels=in_channels
        self.outchannels=out_channels
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(self.inchannels,self.outchannels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    def __init__(self,in_channels,out_channels, bilinear=True):
        super(Up, self).__init__()
        self.inchannels=in_channels
        self.outchannels=out_channels
        self.blinear=bilinear
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.doubleconv = DoubleConv(self.inchannels, self.outchannels, self.inchannels // 2)
        if bilinear==True:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.doubleconv=DoubleConv(self.inchannels,self.outchannels,self.inchannels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self,x1,x2):
        x1=self.up(x1)
        # x1=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)(x1)
        # x1=DoubleConv(self.inchannels, self.outchannels, self.inchannels // 2)(x1)
        diffY=x2.size()[2]-x1.size()[2]
        diffX=x2.size()[3]-x2.size()[3]
        x1=F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv, self).__init__()
        self.inchannels=in_channels
        self.outchannels=out_channels
        self.conv=nn.Conv2d(self.inchannels,self.outchannels,kernel_size=1)
        self.activate=nn.Tanh()
    def forward(self,x):
        x=self.conv(x)
        x=self.activate(x)
        return x


class Phong(nn.Module):
    def __init__(self,n_channels=3, bilinear=False):
        super(Phong, self).__init__()
        self.bilinear=bilinear
        self.double_conv = DoubleConv(n_channels,64)
        self.down1 = Down(64,256)
        self.down2=Down(256,1024)
        self.down3=Down(1024,4096)
        factor=2 if bilinear else 1
        self.up1 =Up(4096,1024,bilinear)
        self.up2=Up(1024,256,bilinear)
        self.up3=Up(256,64,bilinear)
        self.outc=OutConv(64,3)
    def forward(self,x):
        x1=self.double_conv(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)

        x=self.up1(x4,x3)
        x=self.up2(x,x2)
        x=self.up3(x,x1)
        logits=self.outc(x)
        return logits


