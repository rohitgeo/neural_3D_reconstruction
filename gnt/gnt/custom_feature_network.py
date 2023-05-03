import torch
import torch.nn as nn


class ShallowFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1,1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)

        return x, x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)

        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2)

        self.relu = nn.ReLU()

        self.block1 = self.get_block(3, 64)
        self.block2 = self.get_block(64, 128)
        self.block3 = self.get_block(128, 256)
        self.block4 = self.get_block(256, 512)
        self.block5 = self.get_block(512, 1024)

        self.block6 = self.get_block(1024, 512)
        self.block7 = self.get_block(512, 256)

        self.out_conv = nn.Conv2d(256, 32, kernel_size=1)

    def get_block(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3),
                             nn.ReLU(),
                             nn.Conv2d(out_c, out_c, kernel_size=3),
                             nn.ReLU())

    def forward(self, x):
        input_x = x
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.block2(x)
        x = self.max_pool(x)
        x = self.block3(x)
        x = for_upconv2 = self.max_pool(x)
        x = self.block4(x)
        x = for_upconv1 = self.max_pool(x)
        x = self.block5(x)

        x = self.upconv1(x)
        x = self.relu(x)
        x = torch.cat([for_upconv1[:, :, 1:-2, 1:-2], x], dim=1)
        x = self.block6(x)

        x = self.upconv2(x)
        x = self.relu(x)

        x_h, x_w = x.size()[2:4]
        f_h, f_w = for_upconv2.size()[2:4]
        diff_h = f_h - x_h
        diff_w = f_w - x_w

        half_diff_h = diff_h // 2
        half_diff_w = diff_w // 2

        x = torch.cat([for_upconv2[:, :, half_diff_h:-(diff_h - half_diff_h), half_diff_w:-(diff_w - half_diff_w)], x], dim=1)
        x = self.block7(x)

        x = self.out_conv(x)
        x = self.relu(x)

        return x, x





