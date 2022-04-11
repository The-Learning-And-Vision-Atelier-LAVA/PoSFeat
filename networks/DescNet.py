import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)

class ResUNet(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True,
                 coarse_out_ch=128,
                 fine_out_ch=128
                 ):

        super(ResUNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

        # coarse-level conv
        self.conv_coarse = conv(filters[2], coarse_out_ch, 1, 1)

        # decoder
        self.upconv3 = upconv(filters[2], 512, 3, 2)
        self.iconv3 = conv(filters[1] + 512, 512, 3, 1)
        self.upconv2 = upconv(512, 256, 3, 2)
        self.iconv2 = conv(filters[0] + 256, 256, 3, 1)

        # fine-level conv
        self.conv_fine = conv(256, fine_out_ch, 1, 1)
        self.out_channels = [fine_out_ch, coarse_out_ch]

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x = self.firstrelu(self.firstbn(self.firstconv(x)))
        x_first = self.firstmaxpool(x)

        x1 = self.layer1(x_first)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x_coarse = self.conv_coarse(x3) #H/16

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_fine = self.conv_fine(x) #H/4

        return {'global_map':x_coarse, 'local_map':x_fine, 'local_map_small':x_first}

class ResUNetHR(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True,
                 coarse_out_ch=128,
                 fine_out_ch=128
                 ):

        super(ResUNetHR, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

        # coarse-level conv
        self.conv_coarse = conv(filters[2], coarse_out_ch, 1, 1)

        # decoder
        self.upconv3 = upconv(filters[2], 512, 3, 2)
        self.iconv3 = conv(filters[1] + 512, 512, 3, 1)
        self.upconv2 = upconv(512, 256, 3, 2)
        self.iconv2 = conv(filters[0] + 256, 256, 3, 1)
        self.upconv1 = upconv(256,192,3,2)
        self.iconv1 = conv(64 + 192, 256, 3, 1)

        # fine-level conv
        self.conv_fine = conv(256, fine_out_ch, 1, 1)
        self.out_channels = [fine_out_ch, coarse_out_ch]

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x_first1 = self.firstrelu(self.firstbn(self.firstconv(x)))
        x_first = self.firstmaxpool(x_first1)

        x1 = self.layer1(x_first)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x_coarse = self.conv_coarse(x3) #H/16

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x = self.upconv1(x)
        x = self.skipconnect(x_first1, x)
        x = self.iconv1(x)

        x_fine = self.conv_fine(x) #H/2

        return {'global_map':x_coarse, 'local_map':x_fine, 'local_map_small':x_first1}

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)