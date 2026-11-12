import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SkinWithCam(nn.Module):
    """The skin model, also returning one 7x7 activation map per class."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        feats = self.net.features(x)
        logits = self.net.classifier(torch.flatten(self.net.avgpool(feats), 1))
        weight = self.net.classifier[1].weight
        cams = F.conv2d(feats, weight[:, :, None, None])
        return logits, cams


net = models.efficientnet_b0(weights=None)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 7)
net.load_state_dict(torch.load("models/skin_model.pt", map_location="cpu"))
model = SkinWithCam(net).eval()

torch.onnx.export(
    model, (torch.randn(1, 3, 224, 224),), "models/skin_model.onnx",
    input_names=["image"], output_names=["logits", "cams"], external_data=False,
)
print("Saved models/skin_model.onnx")