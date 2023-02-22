import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, filters, strides=(2, 2, 2)):
        super(ConvBlock, self).__init__()
        filters0, filters1, filters2, filters3 = filters
        self.conv_1 = nn.Conv3d(filters0, filters1, kernel_size=1, stride=strides)
        self.bn_1 = nn.BatchNorm3d(filters1)

        self.conv_2 = nn.Conv3d(filters1, filters2, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm3d(filters2)

        self.conv_3 = nn.Conv3d(filters2, filters3, kernel_size=1)
        self.bn_3 = nn.BatchNorm3d(filters3)

        self.conv_res = nn.Conv3d(filters0, filters3, kernel_size=1, stride=strides)
        self.bn_res = nn.BatchNorm3d(filters3)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = nn.ReLU()(out)
        # print('down1:', out.shape)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = nn.ReLU()(out)
        # print('down2:', out.shape)

        out = self.conv_3(out)
        out = self.bn_3(out)
        # print('down3:', out.shape)

        residue = self.conv_res(x)
        residue = self.bn_res(residue)
        # print('down res:', residue.shape)

        final = nn.ReLU()(out + residue)
        return final


class IdentBlock(nn.Module):
    def __init__(self, filters, layer=None):
        super(IdentBlock, self).__init__()
        self.layer = layer
        filter0, filter1, filter2, filter3 = filters
        self.conv_1 = nn.Conv3d(filter0, filter1, kernel_size=1)
        self.bn_1 = nn.BatchNorm3d(filter1)

        self.conv_2 = nn.Conv3d(filter1, filter2, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm3d(filter2)

        self.conv_3= nn.Conv3d(filter2, filter3, kernel_size=1)
        self.bn_3 = nn.BatchNorm3d(filter3)

    def forward(self, input):
        out = self.conv_1(input)
        if self.layer is None:
            out = self.bn_1(out)
        out = nn.ReLU()(out)

        out = self.conv_2(out)
        if self.layer is None:
            out = self.bn_2(out)
        out = nn.ReLU()(out)

        out = self.conv_3(out)
        if self.layer is None:
            out = self.bn_3(out)

        out = nn.ReLU()(out + input)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, filters, stride=(1, 1, 1), size=(2, 2, 2), padding=1, layer=None):
        super(UpConvBlock, self).__init__()
        filters0, filters1, filters2, filters3 = filters
        self.layer = layer
        self.up_sample = nn.Upsample(scale_factor=size)
        self.conv_1 = nn.Conv3d(filters0, filters1, 1, stride)
        self.bn_1 = nn.BatchNorm3d(filters1)

        self.conv_2 = nn.Conv3d(filters1, filters2, 3, padding=padding)
        self.bn_2 = nn.BatchNorm3d(filters2)

        self.conv_3 = nn.Conv3d(filters2, filters3, 1)
        self.bn_3 = nn.BatchNorm3d(filters3)

        self.short_up_sample = nn.Upsample(scale_factor=size)
        self.short_conv = nn.Conv3d(filters0, filters3, 1, stride)
        self.short_bn = nn.BatchNorm3d(filters3)

    def forward(self, input):
        out = self.up_sample(input)
        out = self.conv_1(out)
        if self.layer is None:
            out = self.bn_1(out)
        out = nn.ReLU()(out)

        out = self.conv_2(out)
        if self.layer is None:
            out = self.bn_2(out)
        out = nn.ReLU()(out)

        out = self.conv_3(out)
        if self.layer is None:
            out = self.bn_3(out)

        shortcut = self.short_up_sample(input)
        # print(shortcut.shape)
        shortcut = self.short_conv(shortcut)
        if self.layer is None:
            shortcut = self.short_bn(shortcut)

        # print(out.shape, shortcut.shape)
        out = nn.ReLU()(out + shortcut)
        return out

class PUResNet(nn.Module):
    def __init__(self):
        super(PUResNet, self).__init__()
        f = 18
        self.conv_block_1 = ConvBlock(filters=[18, f, f, f], strides=(1, 1, 1))
        self.ident_block_1a = IdentBlock(filters=[f, f, f, f])
        self.ident_block_1b = IdentBlock(filters=[f, f, f, f])
        self.conv_block_2 = ConvBlock(filters=[f, f * 2, f * 2, f * 2], strides=(2, 2, 2))
        self.ident_block_2a = IdentBlock(filters=[f * 2, f * 2, f * 2, f * 2])
        self.ident_block_2b = IdentBlock(filters=[f * 2, f * 2, f * 2, f * 2])
        self.conv_block_3 = ConvBlock(filters=[f * 2, f * 4, f * 4, f * 4], strides=(2, 2, 2))
        self.ident_block_3a = IdentBlock(filters=[f * 4, f * 4, f * 4, f * 4])
        self.ident_block_3b = IdentBlock(filters=[f * 4, f * 4, f * 4, f * 4])
        self.conv_block_4 = ConvBlock(filters=[f * 4, f * 8, f * 8, f * 8], strides=(3, 3, 3))
        self.ident_block_4a = IdentBlock(filters=[f * 8, f * 8, f * 8, f * 8])
        self.ident_block_4b = IdentBlock(filters=[f * 8, f * 8, f * 8, f * 8])

        self.conv_block_5 = ConvBlock(filters=[f * 8, f * 16, f * 16, f * 16], strides=(3, 3, 3))
        self.ident_block_5a = IdentBlock(filters=[f * 16, f * 16, f * 16, f * 16])

        self.up_conv_block_1 = UpConvBlock(filters=[f * 16, f * 16, f * 16, f * 16], size=(3, 3, 3), padding=1)
        self.ident_block_1 = IdentBlock(filters=[f * 16, f * 16, f * 16, f * 16])

        self.up_conv_block_2 = UpConvBlock(filters=[f * 8 + f * 16, f * 8, f * 8, f * 8], size=(3, 3, 3), stride=(1, 1, 1))
        self.ident_block_2 = IdentBlock(filters=[f * 8, f * 8, f * 8, f * 8])
        self.up_conv_block_3 = UpConvBlock(filters=[f * 4 + f * 8, f * 4, f * 4, f * 4], size=(2, 2, 2), stride=(1, 1, 1))
        self.ident_block_3 = IdentBlock(filters=[f * 4, f * 4, f * 4, f * 4])
        self.up_conv_block_4 = UpConvBlock(filters=[f * 2 + f * 4, f * 2, f * 2, f * 2], size=(2, 2, 2), stride=(1, 1, 1))
        self.ident_block_4 = IdentBlock(filters=[f * 2, f * 2, f * 2, f * 2])
        self.conv_5 = nn.Conv3d(f + f*2, 1, 1)

    def forward(self, inputs):
        x = self.conv_block_1(inputs)
        x = self.ident_block_1a(x)
        x1 = self.ident_block_1b(x)
        # print('down11', x.shape, x1.shape)  # [2, 18, 36, 36, 36]) torch.Size([2, 18, 36, 36, 36]
        x = self.conv_block_2(x)
        x = self.ident_block_2a(x)
        x2 = self.ident_block_2b(x)
        # print('down22', x.shape, x2.shape) # [2, 36, 18, 18, 18]) torch.Size([2, 36, 18, 18, 18]
        x = self.conv_block_3(x)
        x = self.ident_block_3a(x)
        x3 = self.ident_block_3b(x)
        # print('down33', x.shape, x3.shape) # [2, 72, 9, 9, 9]) torch.Size([2, 72, 9, 9, 9])
        x = self.conv_block_4(x)
        x = self.ident_block_4a(x)
        x4 = self.ident_block_4b(x)
        # print('down44', x.shape, x4.shape) # [2, 144, 3, 3, 3]) torch.Size([2, 144, 3, 3, 3])

        x = self.conv_block_5(x)
        x = self.ident_block_5a(x)
        # print('mm', x.shape) # [2, 288, 1, 1, 1])

        x = self.up_conv_block_1(x)
        x = self.ident_block_1(x)
        # print('up44', x.shape)  # [2, 288, 3, 3, 3])
        x = self.up_conv_block_2(torch.cat((x, x4), dim=1))
        x = self.ident_block_2(x)
        # print('up33', x.shape)  # [2, 144, 9, 9, 9])
        x = self.up_conv_block_3(torch.cat((x, x3), dim=1))
        x = self.ident_block_3(x)
        # print('up22', x.shape)  # [2, 72, 18, 18, 18])
        x = self.up_conv_block_4(torch.cat((x, x2), dim=1))
        x = self.ident_block_4(x)
        # print('up11', x.shape)  # [2, 36, 36, 36, 36]
        x = self.conv_5(torch.cat((x, x1), dim=1))
        x = nn.Sigmoid()(x)  # [2, 1, 36, 36, 36]

        return x


if __name__ == '__main__':
    xx = torch.rand(size=(80, 18, 36, 36, 36))
    # xx = torch.rand(size=(2, 14, 65, 65, 65))
    model = PUResNet()
    out = model(xx)
    print('return:', out.shape)
