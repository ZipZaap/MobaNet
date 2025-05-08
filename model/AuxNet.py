import numpy as np

import torch 
import torch.nn as nn

from configs import CONF

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        x = self.convblock(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        self.drop = nn.Dropout(CONF.SEG_DROPOUT)

    def forward(self, x):
        skip = self.conv(x)
        x = self.drop(skip)
        x = self.pool(x)
        return x, skip
    

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)
        self.drop = nn.Dropout(CONF.SEG_DROPOUT)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        x = self.drop(x)
        return x


class EncBody(nn.Module):
    def __init__(self, ch):
        super().__init__()
        """Encoder"""
        self.enc1 = EncoderBlock(CONF.NUM_CHANNELS, ch[0])
        self.enc2 = EncoderBlock(ch[0], ch[1])
        self.enc3 = EncoderBlock(ch[1], ch[2])
        self.enc4 = EncoderBlock(ch[2], ch[3])

        """ Bottleneck """
        self.bneck = ConvBlock(ch[3], ch[4])

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        x = self.bneck(x)
        return x, (skip1, skip2, skip3, skip4)
    

class ClsHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        
        in_c, out_c = ch[4], CONF.CLS_CLASSES
        self.gap = nn.AdaptiveAvgPool2d(1) 
        self.flat = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(in_c, in_c//2),
            nn.BatchNorm1d(in_c//2),
            nn.ReLU(),
            nn.Dropout(CONF.CLS_DROPOUT),
            nn.Linear(in_c//2, in_c//4),
            nn.BatchNorm1d(in_c//4),
            nn.ReLU(),
            nn.Dropout(CONF.CLS_DROPOUT),
            nn.Linear(in_c//4, out_c)
        )

    def forward(self, x):
        x = self.gap(x)
        x = self.flat(x)
        x = self.dense(x)
        return x

class SegHead(nn.Module):
    def __init__(self, ch):
        """Decoder"""
        super().__init__()
        self.dec1 = DecoderBlock(ch[4], ch[3])
        self.dec2 = DecoderBlock(ch[3], ch[2])
        self.dec3 = DecoderBlock(ch[2], ch[1])
        self.dec4 = DecoderBlock(ch[1], ch[0])

        """ Segmenter"""
        self.out2 = nn.Conv2d(ch[0], CONF.SEG_CLASSES, kernel_size=1, padding=0)
        
    def forward(self, x, skips):
        skip1, skip2, skip3, skip4 = skips
        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)
        x = self.out2(x)
        return x

    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        fdepth = CONF.FEATURE_DEPTH
        ldepth = CONF.LAYER_DEPTH
        ch = [int(fdepth*(2**i)) for i in range(ldepth)]

        self.encoder = EncBody(ch)
        self.clshead = ClsHead(ch)
        self.seghead = SegHead(ch)
        
    def forward(self, inputs):
        x, skips = self.encoder(inputs)
        cls = self.clshead(x)
        seg = self.seghead(x, skips)
        return seg, cls
 