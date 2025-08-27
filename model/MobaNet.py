from typing import Optional

import torch
import torch.nn as nn

from utils.util import logits_to_lbl, logits_to_msk
from configs.cfgparser  import Config

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
        
        self.bottleneck = ConvBlock(channels[-2], channels[-1])
        
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
    
# ------------------------------
# Classification Head (Optional)
# ------------------------------
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
    
# ------------------------
# MobaNet (Combined Model)
# ------------------------
class MobaNet(nn.Module):
    def __init__(self, 
                 *,
                 model: str,
                 unet_depth: int,
                 conv_depth: int,
                 in_channels: int,
                 seg_classes: int,
                 cls_classes: int,
                 seg_dropout: Optional[float] = 0.0,
                 cls_dropout: Optional[float] = 0.0,
                 cls_threshold: Optional[float] = None,
                 inference: bool = False
                 ):
        """
        Initializes the MobaNet model.

        Args
        ----
            model : str
                Model type

            unet_depth : int
                Number of layers in the encoder/decoder

            conv_depth : int
                Depth Conv2d block in 1st layer (doubles with each layer).

            in_channels : int
                Number of input channels.

            seg_classes : int
                Number of segmentation classes.

            cls_classes : int
                Number of classification classes.

            seg_dropout : float
                Dropout rate for segmentation layers.

            cls_dropout : float
                Dropout rate for classification layers.

            cls_threshold : float
                Classification threshold for filtering predictions.

            inference : bool
                Whether the model is in inference mode.
        """

        super().__init__()

        self.model = model
        self.inference = inference
        self.cls_threshold = cls_threshold
        self.boundary_class = seg_classes

        enc_ch = [conv_depth * (2 ** i) for i in range(unet_depth)]
        dec_ch = enc_ch[::-1]

        enc_ch = [in_channels] + enc_ch
        dec_ch = dec_ch + [seg_classes]

        self.encoder = Encoder(enc_ch, seg_dropout)
        self.decoder = Decoder(dec_ch, seg_dropout)
        self.classifier = Classifier(enc_ch[-1], cls_classes, cls_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass for the MobaNet model.

        Args
        ----
            x : torch.Tensor (B, C, H, W)
                Input tensor.

        Returns
        -------
            dict[str, torch.Tensor]
                Dictionary containing the output tensors.
        """
        if self.inference:

            B, C, H, W = x.shape
            x, skips = self.encoder(x)
            
            if 'UNet' in self.model:
                # feed the input through the decoder
                seg_logits = self.decoder(x, skips)

                # convert logits to segmentation mask; (B, C, H, W) → (B, 1, H, W)
                seg_mask = logits_to_msk(seg_logits, 'argmax')

                return {'seg': seg_mask}

            else:
                # feed the input through the classifier
                cls_logits = self.classifier(x)

                # convert logits to predicted class labels; (B, C) → (B,)
                lbls = logits_to_lbl(cls_logits, self.cls_threshold)
                
                # create an empty mask tensor & broadcast labels; (B, 1, 1, 1) → (B, 1, H, W)
                seg_mask = torch.zeros((B, 1, H, W), dtype = torch.long, device=x.device)
                seg_mask[:] = lbls[:, None, None, None]

                # run segmentation head only for images belonging to boundary class.
                boundary = (lbls == self.boundary_class)
                if boundary.any():
                    # filter the input and skips for boundary class
                    x = x[boundary]
                    skips = [s[boundary] for s in skips]

                    # feed the input through the segmentation head
                    seg_logits = self.decoder(x, skips)

                    # convert logits to segmentation mask; (B, C, H, W) → (B, 1, H, W)
                    seg_mask[boundary] = logits_to_msk(seg_logits, 'argmax')

                return {'seg': seg_mask}

        else:
            x, skips = self.encoder(x)
            seg_logits = self.decoder(x, skips)

            if 'UNet' in self.model:
                return {'seg': seg_logits}
            
            else: # 'MobaNet' in self.model
                cls_logits = self.classifier(x)
                return {'seg': seg_logits, 'cls': cls_logits}