import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from .clip import *
from .model import VisionTransformer
from utils.network import NetworkManager


vit_parameters = {
    'image_resolution': 224,
    'vision_patch_size': 32,
    'vision_width': 384,
    'vision_layers': 4,
    'vision_heads': 384//16,
    'embed_dim': 512
}

classifier_layels_list = [512, 256] # 512×list[0] -> list[0]×list[1] -> list[1]×list[2] -> ... -> list[-1]×100

def GenTeacher(model_name, device = 'cuda'):
    """
    Generate teacher model from model_name

    Args:
        model_name (str): model name
        device (str): device to load model

    Returns:
        img_encoder (nn.Module): teacher model

    """
    model, _ = load(model_name, device)
    img_encoder = model.visual
    return img_encoder.float()



def GenStudent(vit_parameters = vit_parameters, device = 'cuda'):
    """
    Generate student model from model_name

    Args:
        vit_parameters (dict): parameters for VisionTransformer
        device (str): device to load model

    Returns:
        model (nn.Module): student model

    """
    model = VisionTransformer(
                input_resolution=vit_parameters['image_resolution'],
                patch_size=vit_parameters['vision_patch_size'],
                width=vit_parameters['vision_width'],
                layers=vit_parameters['vision_layers'],
                heads=vit_parameters['vision_heads'],
                output_dim=vit_parameters['embed_dim']
            ).to(device)
    return model

def GenClassifier(input_dim = 512, output_dim = 100, hidden_list=[256], device = 'cuda'):
    """
    Generate classifier model from model_name

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        hidden_list (list): list of hidden layer sizes
        device (str): device to load model

    Returns:
        classifier (nn.Module): classifier model

    """
    net = NetworkManager(input_dim, output_dim)
    classifier = net.create_network(hidden_list).to(device)
    return classifier

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 512)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def GenStudentCNN(device = 'cuda'):
    """
    Generate student model from model_name

    Args:
        vit_parameters (dict): parameters for VisionTransformer
        device (str): device to load model

    Returns:
        model (nn.Module): student model

    """
    model = StudentCNN().to(device)
    return model

if __name__ == '__main__':
    student = GenStudentCNN()
    print("The number of parameters of student model is:")
    print(sum(p.numel() for p in student.parameters()))

    teacher = GenTeacher('ViT-B/32', 'cuda')
    print("The number of parameters of teacher model is:")
    print(sum(p.numel() for p in teacher.parameters()))