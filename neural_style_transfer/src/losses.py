import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models


# Pre-trained VGG19
from torch import Tensor

model=models.vgg19(pretrained=True).features

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class TotalLoss(nn.Module):
    """
    Requires
    p - content image
    a - style image
    x - generated image
    """
    def __init__(self, alpha=1., beta=0.001):
        super(TotalLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()

    def forward(self):
        total_loss = self.alpha * self.content_loss + self.beta * self.style_loss
        return total_loss


class ContentLoss(nn.Module):
    def __init__(self, ):
        super(ContentLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, content_features: Tensor, generated_features: Tensor) -> Tensor:
        return self.mse(content_features, generated_features)




class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self):
        pass