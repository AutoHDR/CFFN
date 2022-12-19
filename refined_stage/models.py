import math
from matplotlib.sankey import UP
# from inplace_abn import InPlaceABN
import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_model import vgg19

from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class SAGAN_Discriminator(nn.Module):
    """SAGAN_Discriminator, Auxiliary Classifier."""

    def __init__(self, conv_dim=64):
        super(SAGAN_Discriminator, self).__init__()
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        curr_dim = curr_dim*2
            
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out #out.squeeze(), p1, p2

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1, bn_flag=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn_flag = bn_flag

    def forward(self, x):
        x = self.conv(x)
        if self.bn_flag:
            x = self.bn(x)
        return x
        
class NormalEncoder_FPN(nn.Module):
    """
    NormalEncoder_FPN output 3 levels of features using a FPN structure
    """
    def __init__(self, dim=3, out_dim=512):
        super(NormalEncoder_FPN, self).__init__()

        ChanFeatures = 64
        self.conv0 = nn.Sequential(
                        ConvBN(dim, ChanFeatures, 3, 1, 1, bn_flag=True),
                        ConvBN(ChanFeatures, ChanFeatures, 3, 1, 1, bn_flag=True))

        self.conv1 = nn.Sequential(
                        ConvBN(ChanFeatures, ChanFeatures, 5, 2, 2, bn_flag=True),
                        ConvBN(ChanFeatures, ChanFeatures, 3, 1, 1, bn_flag=True),
                        ConvBN(ChanFeatures, ChanFeatures, 3, 1, 1, bn_flag=True))

        self.conv2 = nn.Sequential(
                        ConvBN(ChanFeatures, ChanFeatures, 5, 2, 2, bn_flag=True),
                        ConvBN(ChanFeatures, ChanFeatures, 3, 1, 1, bn_flag=True),
                        ConvBN(ChanFeatures, ChanFeatures, 3, 1, 1, bn_flag=True))

        self.conv3 = nn.Sequential(
                        ConvBN(ChanFeatures, ChanFeatures, 5, 2, 2, bn_flag=True),
                        ConvBN(ChanFeatures, ChanFeatures, 3, 1, 1, bn_flag=True),
                        ConvBN(ChanFeatures, ChanFeatures, 3, 1, 1, bn_flag=False))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))#, # 1x1
        self.toplayer = nn.Conv2d(ChanFeatures, out_dim, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 128, 512, 512)
        x = self.conv1(x) # (B, 128, 256, 256)
        x = self.conv2(x) # (B, 128, 128, 128)
        x = self.conv3(x) # (B, 128, 64,  64)
        x = self.pool(x)  # (B, 128, 1,   1)
        x = self.toplayer(x) # (B, 512, 1, 1)

        return x



class SDFT(nn.Module):

    def __init__(self, color_dim, channels, kernel_size = 3):
        super().__init__()
        
        # generate global conv weights
        fan_in = channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(color_dim, channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, channels, channels, kernel_size, kernel_size)
        )

    def forward(self, fea, color_style):
        # for global adjustation
        B, C, H, W = fea.size()
        # print(fea.shape, color_style.shape)
        style = self.modulation(color_style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )

        fea = fea.view(1, B * C, H, W)
        fea = F.conv2d(fea, weight, padding=self.padding, groups=B)
        fea = fea.view(B, C, H, W)

        return fea

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class UpDConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        model = [  nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_channels),
                        nn.ReLU(inplace=True) ]
        self.UpDConvModel = nn.Sequential(*model)

    def forward(self, x):
        x = self.UpDConvModel(x)
        return x


class UpBlock(nn.Module):
    

    def __init__(self, color_dim, in_channels, out_channels, kernel_size = 3, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels // 2 + in_channels // 8, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_s = nn.Conv2d(in_channels//2, out_channels, 1, 1, 0)

        # generate global conv weights
        self.SDFT = SDFT(color_dim, out_channels, kernel_size)


    def forward(self, x1, x2, color_style):
        # print(x1.shape, x2.shape, color_style.shape)
        x1 = self.up(x1)
        x1_s = self.conv_s(x1)

        x = torch.cat([x1, x2[:, ::4, :, :]], dim=1)
        x = self.conv_cat(x)
        x = self.SDFT(x, color_style)

        x = x + x1_s

        return x


class ResBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottle_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.bottle_conv(x)
        x = self.double_conv(x) + x
        return x / math.sqrt(2)


class Down(nn.Module):
    """Downscaling with stride conv then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            # DoubleConv(in_channels, out_channels)
            ResBlock(in_channels, out_channels)
        )  

    def forward(self, x):
        x = self.main(x)
        return x

class Down(nn.Module):
    """Downscaling with stride conv then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            # DoubleConv(in_channels, out_channels)
            ResBlock(in_channels, out_channels)
        )
        
    def forward(self, x):
        x = self.main(x)
        return x

class NormalRefNet(nn.Module):
    ### this model output is ab
    def __init__(self, n_channels=1, n_classes=3, bilinear=True):
        super(NormalRefNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        self.up1 = UpBlock(256, 512, 256 // factor, 3, bilinear)
        self.up2 = UpBlock(256, 256, 128 // factor, 3, bilinear)
        self.up3 = UpBlock(256, 128, 64 // factor, 5, bilinear)
        self.up4 = UpBlock(256, 64, 32, 5, bilinear)
        self.outc = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.Tanh()
        )

    def forward(self, x):
        x_color = x[1] # [B, 512, 1, 1]
        x1 = self.inc(x[0]) # [B, 64, 256, 256]
        x2 = self.down1(x1) # [B, 128, 128, 128]
        x3 = self.down2(x2) # [B, 256, 64, 64]
        x4 = self.down3(x3) # [B, 512, 32, 32]
        x5 = self.down4(x4) # [B, 512, 16, 16]

        x6 = self.up1(x5, x4, x_color) # [B, 256, 32, 32]
        x7 = self.up2(x6, x3, x_color) # [B, 128, 64, 64]
        x8 = self.up3(x7, x2, x_color) # [B, 64, 128, 128]
        x9 = self.up4(x8, x1, x_color) # [B, 64, 256, 256]
        x_ab = self.outc(x9)

        return x_ab