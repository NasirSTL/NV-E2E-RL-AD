import torch
import torchvision
import torch.nn as nn


class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=True)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4
    

class featExtractor(torch.nn.Module):
    def __init__(self, ch=64):
        super(featExtractor,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ch),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ch),
            nn.MaxPool2d(2),
            nn.Conv2d(ch, ch*2, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ch*2),
            nn.MaxPool2d(2),
            nn.Conv2d(ch*2, ch*4, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ch*4),
            nn.MaxPool2d(2),
            nn.Conv2d(ch*4, ch*8, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ch*8),
            nn.MaxPool2d(2),
        )

    def forward(self,x):
        return self.model(x)