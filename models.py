# models.py

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    








# Baseline U-Net architecture

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=23, features=[16, 32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Decoder part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)
    



#U-Net with Feature-wise Linear Modulation (FiLM) at the bottleneck

class UNetWithFiLM(UNet):
    def __init__(self, in_channels=3, num_classes=23, features=[16, 32, 64, 128], feature_len=30, film_hidden_dim=128):
        super().__init__(in_channels=in_channels, num_classes=num_classes, features=features)
        bottleneck_channels = features[-1]
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_len, film_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden_dim, bottleneck_channels * 2) # Predict gamma and beta
        )
        self.bottleneck = DoubleConv(bottleneck_channels, features[-1] * 2)

    def forward(self, x, ft):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # FiLM modulation
        film_params = self.feature_processor(ft)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.view(gamma.size(0), -1, 1, 1)
        beta = beta.view(beta.size(0), -1, 1, 1)
        x = gamma * x + beta
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)





#U-Net that concatenates processed patch-based features at the input

class UNetWithPatchFeatures(UNet):
    def __init__(self, in_channels=3, num_classes=23, features=[16, 32, 64, 128], feature_len=30, patch_hidden_dim=64):
        # The input channels to the first conv layer must be increased to accommodate the feature map
        super().__init__(in_channels=in_channels + features[0], num_classes=num_classes, features=features)
        self.patch_processor = nn.Sequential(
            nn.Conv2d(feature_len, patch_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(patch_hidden_dim, features[0], kernel_size=1)
        )

    def forward(self, x, ft_map):
        processed_ft = self.patch_processor(ft_map)
        processed_ft = TF.resize(processed_ft, size=x.shape[2:])
        x = torch.cat([x, processed_ft], dim=1)
        # After concatenation, pass to the original UNet's forward method
        return super().forward(x)