from torchvision import models
from torch import nn
import torch


class VGGFasttextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        self.vgg16.classifier = nn.Linear(25088, num_classes)

        self.fc1 = nn.Linear(25088, 100)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def apply_vgg(self, image):
        with torch.no_grad():
            features = self.vgg16.avgpool(self.vgg16.features(image))
            features = torch.squeeze(features)
            return features

    def forward(self, image, fasttext_features):
        vgg_features = self.apply_vgg(image)

        all_vgg_features_tensor = torch.flatten(vgg_features)
        X = self.fc1(torch.relu(all_vgg_features_tensor))
        X = torch.cat((X, fasttext_features))
        X = self.fc2(torch.relu(X))
        return self.fc3(torch.relu(X))
