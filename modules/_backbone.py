import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import *


class ExtractorLayer(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            dropout: float=0.0, 
            halve_w: bool=False, 
            halve_h: bool=False):
        
        super(ExtractorLayer, self).__init__()
        
        if out_channels % 2 == 0:
            out, res_out = [out_channels // 2]  * 2
        else:
            res_out = out_channels // 2 
            out = out_channels - res_out
        w_stride = 2 if halve_w else 1
        h_stride = 2 if halve_h else 1
        
        self._layer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 7), stride=(1, w_stride), padding=(1, 7//2)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, out, kernel_size=(3, 7), stride=(h_stride, 1), padding=(1, 7//2)),
            nn.InstanceNorm2d(out),
            nn.Dropout(dropout),
        )
        if not (h_stride or w_stride):
            self._res_layer = nn.Identity()
        else:
            self._res_layer = nn.Conv2d(
                in_channels, res_out, kernel_size=(1, 1), stride=(h_stride, w_stride)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = self._layer(x)
        x2: torch.Tensor = self._res_layer(x)
        output = torch.cat((x1, x2), dim=1)
        return output
    

class ExtractorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, dropout: float=0.0):
        super(ExtractorBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.__make_layers()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for key in self.module_dict.keys():
            x = self.module_dict[key](x)
        return x

    def __make_layers(self):
        in_channels = self.in_channels
        out_channels = 64
        halve_w = False
        halve_h = False
        module_dict = {}
        for i in range(0, self.num_layers):
            if i + 1 == self.num_layers:
                out_channels = self.out_channels
                halve_w = True
            module_dict[f"layer{i}"] = ExtractorLayer(
                in_channels, out_channels, dropout=self.dropout, halve_h=halve_h, halve_w=halve_w
            )
            in_channels = out_channels
            out_channels *= 2
        self.module_dict = nn.ModuleDict(module_dict)
        
    

class CustomBackBone(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            dropout: float=0.0, 
            block_layers: Optional[Iterable[int]]=None
        ):
        super(CustomBackBone, self).__init__()
        
        self.in_channels = in_channels
        self.block_layers = block_layers or [3, 4, 6, 3]
        if len(self.block_layers) != 4:
            raise ValueError("block config must be a list of length = 4")
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=1, padding=7//2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.entry_block = ExtractorBlock(64, 64, 2, dropout=dropout)
        self.block1 = ExtractorBlock(64, 128, self.block_layers[0], dropout=dropout)
        self.block2 = ExtractorBlock(128, 256, self.block_layers[1], dropout=dropout)
        self.block3 = ExtractorBlock(256, 512, self.block_layers[2], dropout=dropout)
        self.block4 = ExtractorBlock(512, 1024, self.block_layers[3], dropout=dropout)
        self.fmap1_ch, self.fmap2_ch, self.fmap3_ch, self.fmap4_ch = [128, 256, 512, 1024]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.first_conv(x)
        x = self.entry_block(x)
        fmap1 = self.block1(x)
        fmap2 = self.block2(fmap1)
        fmap3 = self.block3(fmap2)
        fmap4 = self.block4(fmap3)
        return fmap1, fmap2, fmap3, fmap4
    

class ResNetBackBone(ResNet):
    def __init__(
        self, 
        in_channels: int, 
        dropout: float=0.0, 
        block: Union[str, Type]=BasicBlock, 
        block_layers: Optional[Iterable[int]]=None
    ):
        if isinstance(block, str):
            block = getattr(resnet, block)
        super(ResNetBackBone, self).__init__(block=block, layers=block_layers or [3, 4, 6, 3])
        self.in_channels = in_channels  
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.dropout = nn.Dropout(dropout)

        if block == BasicBlock:
            self.fmap1_ch, self.fmap2_ch, self.fmap3_ch, self.fmap4_ch = [64, 128, 256, 512]
        elif block == Bottleneck:
            self.fmap1_ch, self.fmap2_ch, self.fmap3_ch, self.fmap4_ch = [256, 512, 1024, 2048]
        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        fmap1 = self.layer1(x)
        fmap2 = self.layer2(fmap1)
        fmap3 = self.layer3(fmap2)
        fmap4 = self.layer4(fmap3)
        return fmap1, fmap2, fmap3, fmap4