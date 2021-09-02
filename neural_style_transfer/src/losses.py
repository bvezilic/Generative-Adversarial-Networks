from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

# Pre-trained VGG19
from torch import Tensor

model = models.vgg19(pretrained=True).features


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class TotalLoss(nn.Module):
    def __init__(self, content_features: Tensor, style_features: Tensor, alpha: float = 1., beta: float = 1000.):
        super(TotalLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.content_loss = ContentLoss(content_features)
        self.style_loss = StyleLoss(style_features)

    def forward(self):
        total_loss = self.alpha * self.content_loss + self.beta * self.style_loss
        return total_loss


class ContentLoss(nn.Module):
    def __init__(self, content_features: Tensor):
        super(ContentLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.content_features = content_features

    def forward(self, input_features: Tensor) -> Tensor:
        return self.mse(input_features, self.content_features)


class StyleLoss(nn.Module):
    def __init__(self, style_features: List[Tensor]):
        super(StyleLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.style_features = style_features
        self.style_gram_matrix = torch.cat([gram_matrix(style_feature) for style_feature in self.style_features])

    def forward(self, input_features: List[Tensor]) -> Tensor:
        assert len(input_features) == len(self.style_gram_matrix), \
            f"Mismatched lengths of features! {len(input_features)} != {len(self.style_features)}"

        return self.mse



def gram_matrix(feature_maps: Tensor, normalize: bool = False) -> Tensor:
    B, C, H, W = feature_maps.size()

    assert B == 1, "Batch size must be 1!"

    features = feature_maps.view(B * C, H * W)
    g = torch.mm(features, features.t())

    if normalize:
        g = g / g.numel()

    return g
