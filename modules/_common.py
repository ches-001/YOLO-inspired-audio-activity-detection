import torch
import torch.nn as nn
from typing import *


class ConvBNorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int=1, 
            padding: Optional[int]=None,
            activation: Optional[Type]=None
        ):
        super(ConvBNorm, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=(1, stride), 
            padding=padding
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = None
        if activation:
            self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class BiCModule(nn.Module):
    def __init__(
            self, 
            c1_in_channels1: int, 
            c0_in_channels1: int, 
            p2_in_channels: int, 
            out_channels: int, 
            e: float=0.5
        ):
        super(BiCModule, self).__init__()

        c_h = int(out_channels * e)
        self.upsample_layer = nn.Upsample(scale_factor=(1, 2), mode="bilinear")
        self.down_sample = nn.Upsample(scale_factor=(1, 0.5), mode="bilinear")
        self.conv_c1 = ConvBNorm(c1_in_channels1, c_h, kernel_size=1)
        self.conv_c0 = ConvBNorm(c0_in_channels1, c_h, kernel_size=1)
        self.conv_out = ConvBNorm(c_h+c_h+p2_in_channels, out_channels, kernel_size=1)

    def forward(self, c1: torch.Tensor, c0: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        c1 = self.conv_c1(c1)
        c0 = self.down_sample(self.conv_c0(c0))
        p2 = self.upsample_layer(p2)
        output = torch.cat((c1, c0, p2), dim=1)
        output = self.conv_out(output)
        return output
    

class CSPSPPFModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, e: float=0.5, pool_kernel_size: int=5):
        super(CSPSPPFModule, self).__init__()

        c_h = int(out_channels * e)
        self.conv_1_3_4 = nn.Sequential(
            ConvBNorm(in_channels, c_h, kernel_size=1),
            ConvBNorm(c_h, c_h, kernel_size=3),
            ConvBNorm(c_h, c_h, kernel_size=1)
        )
        self.conv2 = ConvBNorm(in_channels, c_h, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2)
        self.conv5 = ConvBNorm(c_h*4, c_h, kernel_size=1)
        self.conv6 = ConvBNorm(c_h, c_h, kernel_size=3)
        self.conv7 = ConvBNorm(c_h*2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_1_3_4(x)
        y1 = self.conv2(x)
        x_p1 = self.pool(x1)
        x_p2 = self.pool(x_p1)
        x_p3 = self.pool(x_p2)
        x1 = torch.cat((x1, x_p1, x_p2, x_p3), dim=1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x_out = torch.cat((x1, y1), dim=1)
        x_out = self.conv7(x_out)
        return x_out


class MultiScaleFmapModule(nn.Module):
    def __init__(
        self, 
        fmap1_channels: int,
        fmap2_channels: int,
        fmap3_channels: int,
        fmap4_channels: int,
        out_channels: int,
    ):
        super(MultiScaleFmapModule, self).__init__()

        c_h = 64
        self.cspsppf = CSPSPPFModule(fmap4_channels, c_h)
        self.bic2 = BiCModule(fmap2_channels, fmap1_channels, c_h, c_h)
        self.bic3 = BiCModule(fmap3_channels, fmap2_channels, c_h, c_h)
        self.rep_block2_1 = ConvBNorm(c_h, out_channels, kernel_size=3)
        self.rep_block3_1 = ConvBNorm(c_h, c_h, kernel_size=3)
        self.rep_block3_2 = ConvBNorm(c_h*2, out_channels, kernel_size=3)
        self.rep_block4_1 = ConvBNorm(c_h*2, out_channels, kernel_size=3)
        self.identity = nn.Identity()
        self.conv2_downsample = ConvBNorm(out_channels, c_h, kernel_size=3, stride=2)
        self.conv3_downsample = ConvBNorm(out_channels, c_h, kernel_size=3, stride=2)
    
    def forward(
        self, 
        fmap1: torch.Tensor, 
        fmap2: torch.Tensor, 
        fmap3: torch.Tensor, 
        fmap4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        fmap1 = nn.functional.adaptive_avg_pool2d(fmap1, output_size=(1, fmap1.shape[-1]))
        fmap2 = nn.functional.adaptive_avg_pool2d(fmap2, output_size=(1, fmap2.shape[-1]))
        fmap3 = nn.functional.adaptive_avg_pool2d(fmap3, output_size=(1, fmap3.shape[-1]))
        fmap4 = nn.functional.adaptive_avg_pool2d(fmap4, output_size=(1, fmap4.shape[-1]))
        p4 = self.cspsppf(fmap4)
        p3 = self.rep_block3_1(self.bic3(fmap3, fmap2, p4))
        p2 = self.rep_block2_1(self.bic2(fmap2, fmap1, p3))
        n2 = self.identity(p2)
        n3 = self.rep_block3_2(torch.cat((p3, self.conv2_downsample(n2)), dim=1))
        n4 = self.rep_block4_1(torch.cat((p4, self.conv3_downsample(n3)), dim=1))
        n2 = n2.squeeze(dim=2).permute(0, 2, 1)
        n3 = n3.squeeze(dim=2).permute(0, 2, 1)
        n4 = n4.squeeze(dim=2).permute(0, 2, 1)
        return n2, n3, n4