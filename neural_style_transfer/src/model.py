import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = models.vgg19(pretrained=True)

    def forward(self):
        pass

