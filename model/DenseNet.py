import torch 
import torch.nn as nn
import torch.nn.functional as F
    

class Conv(nn.Module):
    def __init__(self, in_c, out_c, act = True, drop = 0.2):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('convolution', nn.Conv2d(in_c, out_c, kernel_size=3, stride = 1, padding=1))
        if act == True:
            self.conv.add_module('activation', nn.ReLU())
        self.conv.add_module('nomralization', nn.BatchNorm2d(out_c))
        self.conv.add_module('dropout', nn.Dropout(drop))
        self.conv.add_module('pooling', nn.MaxPool2d((2, 2)))
        
    def forward(self, x):
        x = self.conv(x)        
        return x
    
class FC(nn.Module):
    def __init__(self, in_dims, out_dims, act = True, drop = 0.5):
        super().__init__()
        self.fc = nn.Sequential()
        self.fc.add_module('linear', nn.Linear(in_dims, out_dims))
        if act == True:
            self.fc.add_module('activation', nn.ReLU())
        self.fc.add_module('dropout', nn.Dropout(drop))
    
    def forward(self, x):
        x = self.fc(x)
        return x
    
class DenseNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        # Convolutional layers
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.conv1 = Conv(1,16, act = True, drop = 0.1)
        self.conv2 = Conv(16,32, act = True, drop = 0.1)
        self.conv3 = Conv(32,64, act = True, drop = 0.1)
        self.conv4 = Conv(64,128, act = True, drop = 0.1)
        self.conv5 = Conv(128,16, act = True, drop = 0.1)
        
        # Fully connected head
        self.fc1 = FC(4096, 1024, act = True, drop = 0.2)
        self.fc2 = FC(1024, 3, act = False, drop = 0.2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = torch.flatten(x,1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    
        
        