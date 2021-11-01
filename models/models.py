import torch.nn as nn
from torchvision import models


def FeatureGenerator(pretrained: bool = True):
    model = models.resnet50(pretrained=pretrained)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    return feature_extractor


class IdClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 2048,
        hidden_size: int = 512,
    ):
        super(IdClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class DomainClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 2048,
        hidden_size_first: int = 1000,
        hidden_size_second: int = 500,
        num_classes: int = 2,
    ):
        super(DomainClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size_first)
        self.fc2 = nn.Linear(hidden_size_first, hidden_size_second)
        self.fc3 = nn.Linear(hidden_size_second, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
