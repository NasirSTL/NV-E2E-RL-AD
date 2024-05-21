import torch
import torch.nn as nn

from .dis_resblock import DisBlock, OptimizedDisBlock


class Discriminator(nn.Module):
    def __init__(self, latent_dim=100, state_dim=5, out_ch=1):
        super(Discriminator, self).__init__()

        self.ch = 64
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.x1 = OptimizedDisBlock(out_ch, self.ch)
        self.c1 = OptimizedDisBlock(self.state_dim, self.ch)
        self.block1 = DisBlock(self.ch * 2, self.ch * 2, downsample=True)
        self.block2 = DisBlock(self.ch * 2, self.ch * 4, downsample=True)
        self.block3 = DisBlock(self.ch * 4, self.ch * 8, downsample=True)
        self.block4 = DisBlock(self.ch * 8, self.ch * 8, downsample=False)
        self.linear = nn.Linear(self.ch * 8, 1, bias=False)
        self.linear = nn.utils.spectral_norm(self.linear)

        # self.x_layers = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
        #     nn.BatchNorm2d(32, momentum=0.9),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

        # self.c_layers = nn.Sequential(
        #     nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

        # self.backbone = nn.Sequential(
        #     nn.Conv2d(192, 256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0),
        #     nn.Sigmoid()
        # )

    def forward(self, x, c):
        # # if x is a 3D tensor, add a channel dimension
        # if x.dim() == 3:
        #     x = x.unsqueeze(0) 
        # x = self.x_layers(x)
        # c = c.view(-1, 2, 1, 1)
        # c = c.expand(-1, 2, 32, 32)
        # c = self.c_layers(c)
        # x = torch.cat([x, c], dim=1)
        # x = self.backbone(x)

        if x.dim() == 3:
            x = x.unsqueeze(0) 
        c = c.view(-1, self.state_dim, 1, 1)
        c = c.expand(-1, self.state_dim, 128, 128)
        x = self.x1(x)
        c = self.c1(c)
        x = torch.cat([x, c], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.ReLU()(x)
        # Global average pooling
        x = x.sum(2).sum(2)
        x = self.linear(x)

        return x
