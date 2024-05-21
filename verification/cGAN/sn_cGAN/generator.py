import torch
import torch.nn as nn

from .gen_resblock import GenBlock


class Generator(nn.Module):
    def __init__(self, noise_dim=100, state_dim=5, out_ch=1, version='v1'):
        super(Generator, self).__init__()

        self.z_width = 4
        self.c_width = 4
        self.ch = 64

        self.version = version

        if self.version == 'v1':
            self.z1 = nn.Linear(noise_dim, (self.z_width ** 2) * self.ch * 8)  # latent_dim -> latent image dimension * channel
            self.z2 = GenBlock(self.ch * 8, self.ch * 8, upsample=True)
            self.c1 = nn.Linear(state_dim, (self.c_width ** 2) * self.ch * 8)  # state_dim -> latent image dimension * channel
            self.c2 = GenBlock(self.ch * 8, self.ch * 8, upsample=True)
            self.block1 = GenBlock(self.ch * 8 * 2, self.ch * 8, upsample=True)
            self.block2 = GenBlock(self.ch * 8, self.ch * 4, upsample=True)
            self.block3 = GenBlock(self.ch * 4, self.ch * 2, upsample=True)
            self.block4 = GenBlock(self.ch * 2, self.ch, upsample=True)
        elif self.version == 'v2':
            self.z1 = nn.Linear(noise_dim, (self.z_width ** 2) * self.ch * 4)  # latent_dim -> latent image dimension * channel
            self.z2 = GenBlock(self.ch * 4, self.ch, upsample=True)
            self.c1 = nn.Linear(state_dim, (self.c_width ** 2) * self.ch * 8)  # state_dim -> latent image dimension * channel
            self.c2 = GenBlock(self.ch * 8, self.ch * 16, upsample=True)
            self.block1 = GenBlock(self.ch * 17, self.ch * 16, upsample=True)
            self.block2 = GenBlock(self.ch * 16, self.ch * 8, upsample=True)
            self.block3 = GenBlock(self.ch * 8, self.ch * 4, upsample=True)
            self.block4 = GenBlock(self.ch * 4, self.ch, upsample=True)

        self.final = nn.Sequential(
            nn.BatchNorm2d(self.ch),
            nn.ReLU(),
            nn.Conv2d(self.ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            # nn.Sigmoid()
        )

        # self.z_layer = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=noise_dim, out_channels=200, kernel_size=4, stride=1),
        #     nn.BatchNorm2d(200, momentum=0.9),
        #     nn.LeakyReLU(0.2, True)
        # )
        
        # self.c_layer = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=state_dim, out_channels=312, kernel_size=4, stride=1),
        #     nn.BatchNorm2d(312, momentum=0.9),
        #     nn.LeakyReLU(0.2, True)
        # )

        # self.backbone = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(256, momentum=0.9),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(128, momentum=0.9),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(64, momentum=0.9),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(32, momentum=0.9),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        #     nn.Tanh()
        # )


    def forward(self, z, c):
        # z = z.view(-1, 100, 1, 1)
        # c = c.view(-1, 2, 1, 1)
        # z = self.z_layer(z)
        # c = self.c_layer(c)
        # x = torch.cat([z, c], dim=1)
        # x = self.backbone(x)

        # print(f'generator input shape: {z.shape}, {c.shape}')

        if self.version == 'v1':
            z = self.z1(z).view(-1, self.ch * 8, self.z_width, self.z_width)
            z = self.z2(z)
            c = self.c1(c).view(-1, self.ch * 8, self.c_width, self.c_width)
            c = self.c2(c)
        elif self.version == 'v2':
            z = self.z1(z).view(-1, self.ch * 4, self.z_width, self.z_width)
            z = self.z2(z)
            c = self.c1(c).view(-1, self.ch * 8, self.c_width, self.c_width)
            c = self.c2(c)

        # print(f'after linear layers: {z.shape}, {c.shape}')
        x = torch.cat([z, c], dim=1)
        # print(f'after concatenation: {x.shape}')
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final(x)

        # print(f'Generator output shape: {x.shape}')

        return x
