import torch.nn as nn
import timm


class Predictor(nn.Module):
    """ The header to predict age (regression branch) """

    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features//4, kernel_size=3, padding=3//2), 
            nn.BatchNorm2d(num_features//4), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(num_features//4, num_features//16, kernel_size=3, padding=3//2), 
            nn.BatchNorm2d(num_features//16), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(num_features//16, num_features//32, kernel_size=3, padding=3//2), 
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(num_features//32, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        return x


class Classifier(nn.Module):
    """ The header to predict gender (classification branch) """

    def __init__(self, num_features, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features//4, kernel_size=3, padding=3//2), 
            nn.BatchNorm2d(num_features//4), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(num_features//4, num_features//16, kernel_size=3, padding=3//2), 
            nn.BatchNorm2d(num_features//16), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(num_features//16, num_features//32, kernel_size=3, padding=3//2), 
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(num_features//32, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.squeeze(-1).squeeze(-1)
        return x


class Model(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=True):
        super().__init__()

        self.backbone = timm.create_model("resnet50", pretrained=timm_pretrained)
        self.predictor = Predictor(self.backbone.num_features)
        self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        age = self.predictor(x)
        gender = self.classifier(x)

        return age, gender
