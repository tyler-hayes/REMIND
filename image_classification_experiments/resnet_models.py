import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet18_StartAt_Layer4_1(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer4_1, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3
        del self.model.layer4[0]

    def forward(self, x):
        out = self.model.layer4(x)
        out = F.avg_pool2d(out, out.size()[3])
        final_embedding = out.view(out.size(0), -1)
        out = self.model.fc(final_embedding)
        return out


class ResNet18_StartAt_Layer4_0(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer4_0, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3

    def forward(self, x):
        out = self.model.layer4(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer3_1(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer3_1, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3[0]

    def forward(self, x):
        out = self.model.layer3(x)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer3_0(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer3_0, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2

    def forward(self, x):
        out = self.model.layer3(x)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer2_1(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer2_1, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2[0]

    def forward(self, x):
        out = self.model.layer2(x)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer2_0(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer2_0, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1

    def forward(self, x):
        out = self.model.layer2(x)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer1_1(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer1_1, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1[0]

    def forward(self, x):
        out = self.model.layer1(x)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer1_0(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer1_0, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1

    def forward(self, x):
        out = self.model.layer1(x)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_FC(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_FC, self).__init__()

        self.model = models.resnet18(pretrained=False)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3
        del self.model.layer4

    def forward(self, x):
        out = F.avg_pool2d(x, x.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class BaseResNet18ClassifyAfterLayer4(nn.Module):
    def __init__(self, num_del=0, num_classes=None):
        super(BaseResNet18ClassifyAfterLayer4, self).__init__()
        self.model = models.resnet18(pretrained=False)
        for _ in range(0, num_del):
            del self.model.layer4[-1]
        if num_classes is not None:
            print("Changing num_classes to {}".format(num_classes))
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet18ClassifyAfterLayer4_0(BaseResNet18ClassifyAfterLayer4):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer4_0, self).__init__(num_del=1, num_classes=num_classes)


class ResNet18ClassifyAfterLayer4_1(BaseResNet18ClassifyAfterLayer4):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer4_1, self).__init__(num_del=0, num_classes=num_classes)