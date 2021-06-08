""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

#from .unet_parts import *
from .search_unet_parts import *

class search_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(search_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)   #b_s, 64, 224, 224
        #b_s, 128, 112, 112
        self.down1 = Cell(64, 64, 32, True, False)
        #b_s, 256, 56, 56
        self.down2 = Cell(64, 128, 64, True, True)
        #b_s, 512, 28, 28
        self.down3 = Cell(128, 256, 128, True, True)
        factor = 2 if bilinear else 1
        #b_s, 512, 14, 14
        self.down4 = Cell(256, 512, 128, True, True)
        #b_s, 256, 28, 28
        self.up1 = Up(512, 1024, 64)
        self.up2 = Up(512, 512, 32)             #b_s, 128, 56, 56
        self.up3 = Up(256, 256, 16)             #b_s, 64, 112, 112
        self.up4 = Up(128, 128, 16)             #b_s, 64, 224, 224
        self.outc = OutConv(64, n_classes)                      #b_s, 1, 224, 224

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1, x1)
        x3 = self.down2(x1, x2)
        x4 = self.down3(x2, x3)
        x5 = self.down4(x3, x4)
        temp0 = self.up1(x5, x5, x4, True)
        temp1 = self.up2(x5, temp0, x3, False)
        temp0, temp1 = temp1, self.up3(temp0, temp1, x2, False)
        temp0, temp1 = temp1, self.up4(temp0, temp1, x1, False)
        logits = self.outc(temp1)
        return logits
        #return temp1
if __name__=="__main__":
    net = search_UNet(n_channels=3, n_classes=1, bilinear=True).cuda()
    image = torch.randn(1, 3, 224, 224).cuda()
    result = net(image)
    print(result.shape)
