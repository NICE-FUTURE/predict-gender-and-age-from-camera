import torch.nn as nn
import timm
from timm.models.layers import Mlp


class Regression(nn.Module):
    """ The header to predict age (regression branch) """

    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.mlp = Mlp(num_features, hidden_features=num_features//2, out_features=num_features//4, drop=0.5)
        self.fc = nn.Linear(num_features//4, num_classes)

    def forward(self, x):
        x = self.mlp(x)
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    """ The header to predict gender (classification branch) """

    def __init__(self, num_features, num_classes=2):
        super().__init__()
        self.mlp = Mlp(num_features, hidden_features=num_features//2, out_features=num_features//4, drop=0.5)
        self.fc = nn.Linear(num_features//4, num_classes)

    def forward(self, x):
        x = self.mlp(x)
        x = self.fc(x)
        return x


class Model(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_arch="swin_small_patch4_window7_224", timm_pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(timm_arch, pretrained=timm_pretrained)
        self.predictor = Regression(self.backbone.num_features)
        self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, L, C
        x = x.mean(dim=1)  # shape: B, C
        age = self.predictor(x)
        gender = self.classifier(x)

        return age, gender
