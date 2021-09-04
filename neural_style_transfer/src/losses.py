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
    def __init__(self, content_features: Tensor, style_features: List[Tensor], alpha: float = 1., beta: float = 1000.):
        super(TotalLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.content_loss = ContentLoss(content_features)
        self.style_loss = StyleLoss(style_features)

    def forward(self, input_content_features: Tensor, input_style_feature: Tensor):
        total_loss = self.alpha * self.content_loss(input_content_features) + \
                     self.beta * self.style_loss(input_style_feature)
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
        self.style_gram_matrix = [gram_matrix(style_feature) for style_feature in self.style_features]

    @property
    def num_layers(self):
        return len(self.style_features)  # This will be constant w

    def forward(self, input_features: List[Tensor]) -> Tensor:
        assert len(input_features) == len(self.style_gram_matrix), \
            f"Mismatched lengths of features! {len(input_features)} != {len(self.style_features)}"

        inputs_gram_matrix = [gram_matrix(inpute_feature) for inpute_feature in input_features]

        style_loss = 0
        for style_gram, input_gram in zip(self.style_gram_matrix, inputs_gram_matrix):
            e_l = self.mse(input_gram, style_gram)
            style_loss += e_l

        return style_loss / self.num_layers


def gram_matrix(feature_maps: Tensor, normalize: bool = False) -> Tensor:
    B, C, H, W = feature_maps.size()

    assert B == 1, f"Batch size must be 1! Got B={B}"

    feature_maps = feature_maps.squeeze(0)  # Remove batch_size
    features = feature_maps.view(C, H * W)
    g = torch.mm(features, features.t())

    if normalize:
        g = g / g.numel()

    return g
