from typing import List, Dict

import torch.nn as nn
import torchvision.models as models
from torch import Tensor


class VGG19(nn.Module):
    def __init__(self,
                 content_layers: List[int],
                 style_layers: List[int]):
        super(VGG19, self).__init__()

        self.model = models.vgg19(pretrained=True).features

        self.content_layers = content_layers
        self.style_layers = style_layers

        self.content_features = {}
        self.style_features = {}

        # Register hooks to obtain various inputs
        for idx, module in enumerate(self.model.children()):
            if idx in self.content_layers:
                module.register_forward_hook(self._get_content_activation(idx))

            if idx in self.style_layers:
                module.register_forward_hook(self._get_style_activation(idx))

    def _get_content_activation(self, idx):
        def hook(module, input, output) -> None:
            self.content_features[idx] = output

        return hook

    def _get_style_activation(self, idx):
        def hook(module, input, output) -> None:
            self.style_features[idx] = output

        return hook

    def clear_features(self):
        self.content_features = {}
        self.style_features = {}

    @staticmethod
    def clone_features(features: Dict[int, Tensor]):
        return {idx: feature.clone() for idx, feature in features.items()}

    def forward(self, input_image: Tensor):
        x = self.model(input_image)

        content_features = self.clone_features(self.content_features)
        style_features = self.clone_features(self.style_features)

        self.clear_features()

        return x, content_features, style_features
