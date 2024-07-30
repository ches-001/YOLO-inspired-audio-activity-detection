import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class ConvBorINorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]]=1, 
            padding: Optional[Union[int, Tuple[int, int]]]=None,
            norm_layer: Type=nn.InstanceNorm2d,
            activation: Optional[Type]=nn.LeakyReLU,
            bias: bool=True,
        ):
        super(ConvBorINorm, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias
        )
        self.norm = norm_layer(out_channels)
        self.activation = None
        if activation:
            if activation == nn.LeakyReLU:
                self.activation = activation(0.2)
            else:
                self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class RepVGGBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            activation: Optional[Type]=nn.LeakyReLU, 
            stride: Union[int, Tuple[int, int]]=1,
            padding: Optional[Union[int, Tuple[int, int]]]=None,
            identity_layer: Type=nn.InstanceNorm2d
        ):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding or 3//2
        self.inference_mode = False

        self.conv3x3 = ConvBorINorm(
            in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=self.padding, bias=False
        )
        self.conv1x1 = ConvBorINorm(
            in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=self.padding-1, bias=False
        )
        if stride == 1 and in_channels == out_channels:
            self.identity = identity_layer(out_channels)
        else:
            self.identity = nn.Identity()
        if activation:
            if activation == nn.LeakyReLU:
                self.activation = activation(0.2)
            else:
                self.activation = activation()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv_reparam"):
            return self.activation(self.conv_reparam(x))
        
        out = self.conv3x3(x) + self.conv1x1(x)
        if not isinstance(self.identity, nn.Identity):
            out = out + self.identity(x)
        if self.activation:
            out = self.activation(out)
        return out
    
    def reparameterize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w3x3, b3x3 = self._merge_conv_bn(self.conv3x3.conv, self.conv3x3.norm)
        w1x1, b1x1 = self._merge_conv_bn(self.conv1x1.conv, self.conv1x1.norm)
        w = w3x3 + F.pad(w1x1, [1, 1, 1, 1])
        b = b3x3 + b1x1
        if not isinstance(self.identity, nn.Identity):
            wI1x1, bI1x1 = self._merge_conv_bn(nn.Identity(), self.identity)
            w = w + F.pad(wI1x1, [1, 1, 1, 1])
            b = b + bI1x1
        return w, b
    
    def _merge_conv_bn(
            self, 
            conv: Union[nn.Conv2d, nn.Identity], 
            bn: nn.BatchNorm2d,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(bn, nn.BatchNorm2d):
            raise RuntimeError(
                f"RepVGGBlock reparameterization only works with nn.BatchNorm2d layers, got {type(bn)} instead"
            )
        if isinstance(conv, nn.Conv2d):
            w = conv.weight
        elif isinstance(conv, nn.Identity):
            input_dim = self.in_channels//self.conv3x3.conv.groups
            w = torch.zeros((self.in_channels, input_dim, 1, 1), device=self.conv3x3.conv.weight.device)
            for i in range(self.in_channels):
                w[i, i % input_dim, 0, 0] = 1
        else: 
            raise RuntimeError
        gamma = bn.weight
        mu = bn.running_mean
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(bn.running_var + eps)
        weight_n = (gamma / std).reshape(-1, *([1]*(len(w.shape)-1))) * w
        bias_n = ((-mu * gamma) / std) + beta
        return weight_n, bias_n
    
    def toggle_inference_mode(self):
        w, b = self.reparameterize()
        self.conv_reparam = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=(3, 3), stride=self.stride, padding=self.padding
        )
        self.conv_reparam.weight.data = w
        self.conv_reparam.bias.data = b
        if hasattr(self, "conv3x3"): self.__delattr__("conv3x3")    
        if hasattr(self, "conv1x1"): self.__delattr__("conv1x1")   
        if hasattr(self, "identity"): self.__delattr__("identity")
        self.inference_mode = True
        

class RepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int=2):
        super(RepBlock, self).__init__()
        self.conv1 = RepVGGBlock(in_channels, out_channels)
        if n > 1:
            self.blocks = nn.Sequential(*[RepVGGBlock(out_channels, out_channels) for i in range(n-1)])
        else:
            self.blocks = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.conv1(x))
    

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
        self.conv_c1 = ConvBorINorm(c1_in_channels1, c_h, kernel_size=1)
        self.conv_c0 = ConvBorINorm(c0_in_channels1, c_h, kernel_size=1)
        self.conv_out = ConvBorINorm(c_h+c_h+p2_in_channels, out_channels, kernel_size=1)

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
            ConvBorINorm(in_channels, c_h, kernel_size=1),
            ConvBorINorm(c_h, c_h, kernel_size=3),
            ConvBorINorm(c_h, c_h, kernel_size=1)
        )
        self.conv2 = ConvBorINorm(in_channels, c_h, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2)
        self.conv5 = ConvBorINorm(c_h*4, c_h, kernel_size=1)
        self.conv6 = ConvBorINorm(c_h, c_h, kernel_size=3)
        self.conv7 = ConvBorINorm(c_h*2, out_channels, kernel_size=1)

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

        c_h = 128
        self.cspsppf = CSPSPPFModule(fmap4_channels, c_h)
        self.bic2 = BiCModule(fmap2_channels, fmap1_channels, c_h, c_h)
        self.bic3 = BiCModule(fmap3_channels, fmap2_channels, c_h, c_h)
        self.rep_block2_1 = RepBlock(c_h, out_channels)
        self.rep_block3_1 = RepBlock(c_h, c_h)
        self.rep_block3_2 = RepBlock(c_h*2, out_channels)
        self.rep_block4_1 = RepBlock(c_h*2, out_channels)
        self.identity = nn.Identity()
        self.conv2_downsample = ConvBorINorm(out_channels, c_h, kernel_size=3, stride=(1, 2))
        self.conv3_downsample = ConvBorINorm(out_channels, c_h, kernel_size=3, stride=(1, 2))
    
    def forward(
        self, 
        fmap1: torch.Tensor, 
        fmap2: torch.Tensor, 
        fmap3: torch.Tensor, 
        fmap4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fmap1.shape[-2] != fmap2.shape[-2] != fmap3.shape[-2] != fmap4.shape[-2]:
            fmap1 = nn.functional.adaptive_avg_pool2d(fmap1, output_size=(1, fmap1.shape[-1]))
            fmap2 = nn.functional.adaptive_avg_pool2d(fmap2, output_size=(1, fmap2.shape[-1]))
            fmap3 = nn.functional.adaptive_avg_pool2d(fmap3, output_size=(1, fmap3.shape[-1]))
            fmap4 = nn.functional.adaptive_avg_pool2d(fmap4, output_size=(1, fmap4.shape[-1]))
        p4 = self.cspsppf(fmap4)
        p3 = self.rep_block3_1(self.bic3(fmap3, fmap2, p4))
        p2 = self.rep_block2_1(self.bic2(fmap2, fmap1, p3))
        n2 = self.identity(p2)
        n3 = self.rep_block3_2(torch.cat((p3, self.conv2_downsample(n2)), dim=1))
        n4 = self.rep_block3_2(torch.cat((p4, self.conv3_downsample(n3)), dim=1))
        n2 = nn.functional.adaptive_avg_pool2d(n2, output_size=(1, n2.shape[-1]))
        n3 = nn.functional.adaptive_avg_pool2d(n3, output_size=(1, n3.shape[-1]))
        n4 = nn.functional.adaptive_avg_pool2d(n4, output_size=(1, n4.shape[-1]))
        n2 = n2.squeeze(dim=2).permute(0, 2, 1)
        n3 = n3.squeeze(dim=2).permute(0, 2, 1)
        n4 = n4.squeeze(dim=2).permute(0, 2, 1)
        return n2, n3, n4