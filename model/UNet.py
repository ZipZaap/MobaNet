import torch 
import torch.nn as nn

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

    def forward(self, inputs):
        x = self.convblock(inputs)
        return x


class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        self.drop = nn.Dropout(0.1)

    def forward(self, inputs):
        skip = self.conv(inputs)
        x = self.drop(skip)
        x = self.pool(x)
        return x, skip
    

class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)
        self.drop = nn.Dropout(0.1)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        x = self.drop(x)
        return x

    
class UNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.cls = conf.NUM_CLASSES
        self.chl = conf.NUM_CHANNELS

        """ Encoder """
        self.enc1 = Encoder(self.chl, 16)
        self.enc2 = Encoder(16, 32)
        self.enc3 = Encoder(32, 64)
        self.enc4 = Encoder(64, 128)
        
        """ Bottleneck """
        self.bneck = ConvBlock(128, 256)

        """ Decoder """
        self.dec1 = Decoder(256, 128)
        self.dec2 = Decoder(128, 64)
        self.dec3 = Decoder(64, 32)
        self.dec4 = Decoder(32, 16)
        
        """ Segmenter"""
        self.out2 = nn.Conv2d(16, self.cls, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        x, skip1 = self.enc1(inputs)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        """ Bottleneck """
        x = self.bneck(x)

        """ Decoder """
        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)
    
        """ Segmenter """
        x = self.out2(x)

        return x