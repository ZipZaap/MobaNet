import torch
import torch.nn as nn
import torch

from configs.config_parser import CONF

# -----------------------------
# Basic Convolutional Block
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        return self.convblock(x)
    
# -------------
# Encoder Block
# -------------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        skip = self.conv(x)
        x = self.drop(skip)
        x = self.pool(x)
        return x, skip
    
# -------------
# Decoder Block
# -------------
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c + out_c, out_c) # 2 x out_c channels (one from upsampling and one from the skip)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.drop(x)
        return x
    
# --------------------------------
# Encoder Body (Downsampling path)
# --------------------------------
class Encoder(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.encoders = nn.ModuleList()

        for i in range(len(channels) - 2):
            self.encoders.append(EncoderBlock(channels[i], channels[i+1], dropout))
        
        self.bottleneck = ConvBlock(channels[-1], channels[-2])
        
    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        x = self.bottleneck(x)
        return x, skips
    
# --------------------------------
# Segmentation Head (Decoder path)
# --------------------------------
class Decoder(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.decoders = nn.ModuleList()

        for i in range(len(channels) - 2):
            self.decoders.append(DecoderBlock(channels[i], channels[i+1], dropout))

        self.out = nn.Conv2d(channels[-2], channels[-1], kernel_size=1)
        
    def forward(self, x, skips):
        for decoder, skip in zip(self.decoders, skips[::-1]):
            x = decoder(x, skip)
        x = self.out(x)
        return x
    
# -----------------------------
# Classification Head (Optional)
# -----------------------------
class Classifier(nn.Module):
    def __init__(self, bottleneck_channels, cls_classes, cls_dropout):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(bottleneck_channels, bottleneck_channels // 2),
            nn.BatchNorm1d(bottleneck_channels // 2),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(bottleneck_channels // 2, bottleneck_channels // 4),
            nn.BatchNorm1d(bottleneck_channels // 4),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(bottleneck_channels // 4, cls_classes)
        )
        
    def forward(self, x):
        x = self.gap(x)
        x = self.flat(x)
        x = self.dense(x)
        return x
    
# ---------------------
# UNet (Combined Model)
# ---------------------
class UNet(nn.Module):
    def __init__(self,
                 model: str,
                 num_layers = CONF.LAYER_DEPTH,
                 l1_depth = CONF.FEATURE_DEPTH,
                 in_channels = CONF.NUM_CHANNELS,
                 seg_classes = CONF.SEG_CLASSES,
                 seg_dropout = CONF.SEG_DROPOUT, 
                 cls_classes = CONF.CLS_CLASSES,
                 cls_dropout = CONF.CLS_DROPOUT):

        super().__init__()

        self.model = model

        enc_ch = [l1_depth * (2 ** i) for i in range(num_layers)]
        dec_ch = enc_ch[::-1]

        enc_ch = [in_channels] + enc_ch
        dec_ch = dec_ch + [seg_classes]

        self.encoder = Encoder(enc_ch, seg_dropout)
        self.decoder = Decoder(dec_ch, seg_dropout)
        self.classifier = Classifier(enc_ch[-1], cls_classes, cls_dropout)

    def forward(self, x):
        x, skips = self.encoder(x)
        if 'UNet' in self.model:
            seg = self.decoder(x, skips)
            return {'seg': seg}
        elif 'AuxNet' in self.model:
            seg = self.decoder(x, skips)
            cls = self.classifier(x)
            return {'seg': seg, 'cls': cls}