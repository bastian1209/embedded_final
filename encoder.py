import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, padding=1, norm_layer=None, is_last=False):
        super(BasicBlock, self).__init__()
        
        self.is_last = is_last
        self.norm_layer = nn.BatchNorm2d if norm_layer == None else norm_layer
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(self.expansion * planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        
        if self.is_last:
            return out, preact
        else:
            return out
        
        
class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, is_last=False):
        super(BottleNeck, self).__init__()
        self.is_last = is_last
        self.norm_layer = nn.BatchNorm2d if norm_layer == None else norm_layer
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = self.norm_layer(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(self.expansion * planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        
        if self.is_last:
            return out, preact
        else:
            return out
        
            
class ResNet(nn.Module):
    def __init__(self, block, num_blocks_list, in_channels=3, norm_layer=None, is_cifar=False, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm_layer = nn.BatchNorm2d if norm_layer == None else norm_layer
        
        if is_cifar:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False), 
                self.norm_layer(self.in_planes), 
                nn.ReLU(inplace=True)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False), 
                self.norm_layer(self.in_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        self.layer1 = self._make_layer(block, 64, num_blocks_list[0])
        self.layer2 = self._make_layer(block, 128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn3.weight, 0.)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.)
        
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, planes, strides[i], self.norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        return out
    
    
def resnet18(**kwargs):
     return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)


model_dict = {
    'resnet18' : [resnet18, 512], 
    'resnet50' : [resnet50, 2048]
}


class MLPhead(nn.Module):
    def __init__(self, in_features, out_features, hidden, method='moco'):
        super(MLPhead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        if (method == 'byol') or (method == 'simclr_w_bn'):
            self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, out_features)
        if method == 'simclr_w_bn':
            self.bn2 = nn.BatchNorm1d(out_features)
            
    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if hasattr(self, 'bn2'):
            x = self.bn2(x)
        
        return x
    
    
class ResNet_SSL(nn.Module):
    def __init__(self, arch_name='resnet50', head='mlp', 
                 encoder_params={"norm_layer" : None, 'is_cifar' : False, 'zero_init_residual' : False}, ssl_feat_dim=128, method='moco'):
        super(ResNet_SSL, self).__init__()
        model_func, feat_dim = model_dict[arch_name]
        self.encoder = model_func(**encoder_params)
        
        hidden = feat_dim
        if method == 'byol':
            ssl_feat_dim = 256
            hidden = feat_dim * 2
            
        if head == 'mlp':
            self.proj_head = MLPhead(feat_dim, ssl_feat_dim, hidden=hidden, method=method)
        elif head == 'linear':
            self.proj_head = nn.Linear(feat_dim, ssl_feat_dim)
            
    def forward(self, x):
        feature = self.encoder(x)
        feature = F.normalize(self.proj_head(feature), dim=1)
        
        return feature
    
    
class ResNet_SL(nn.Module):
    def __init__(self, arch_name='resnet50', encoder_params={"norm_layer" : None, "is_cifar" : False}, num_classes=1000, use_bn=False):
        super(ResNet_SL, self).__init__()
        
        self.use_bn = use_bn
        if encoder_params['is_cifar'] == True:
            num_classes = 10
            
        model_func, feat_dim = model_dict[arch_name]
        self.encoder = model_func(**encoder_params)
        
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim, affine=False)
        
        self.fc = nn.Linear(feat_dim, num_classes) 
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
            
    def forward(self, x):
        if self.use_bn:
            return self.fc(self.bn(self.encoder(x)))
        return self.fc(self.encoder(x))