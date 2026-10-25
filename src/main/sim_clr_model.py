import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


# -------------------------------------------------------
# Encoder Backbone (f in SimCLR paper)
# -------------------------------------------------------
# This is the feature extractor.
# It outputs the representation h = f(x)
#
# Adapted for CIFAR-10 / STL-10 sized images:
# - smaller first conv (3x3, stride 1)
# - remove maxpool
# - remove final classifier
# -------------------------------------------------------
class SimCLRBase(nn.Module):
    def __init__(self, arch="resnet18"):
        super().__init__()

        if arch != "resnet18":
            raise ValueError(f"Unsupported architecture: {arch}")

        backbone = resnet18(weights=None)

        layers = []
        for name, module in backbone.named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )

            if not isinstance(module, (nn.Linear, nn.MaxPool2d)):
                layers.append(module)

        self.encoder = nn.Sequential(*layers)
        self.feature_dim = 512

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return x


# -------------------------------------------------------
# Full SimCLR model = encoder f + projection head g
# -------------------------------------------------------
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, arch="resnet18"):
        super().__init__()

        self.f = SimCLRBase(arch)
        in_dim = self.f.feature_dim

        self.g = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

        self.feature_dim = in_dim
        self.projection_dim = feature_dim

    def encode(self, x, normalize=True):
        feature = self.f(x)
        if normalize:
            feature = F.normalize(feature, dim=1)
        return feature

    def project(self, feature, normalize=True):
        out = self.g(feature)
        if normalize:
            out = F.normalize(out, dim=1)
        return out

    def forward(self, x):
        feature = self.f(x)
        out = self.g(feature)

        feature = F.normalize(feature, dim=1)
        out = F.normalize(out, dim=1)

        return feature, out


# -------------------------------------------------------
# Helper function to build the model
# -------------------------------------------------------
def get_encoder_architecture(args):
    arch = getattr(args, "arch", "resnet18")
    feature_dim = getattr(args, "feature_dim", 128)
    return SimCLR(feature_dim=feature_dim, arch=arch)