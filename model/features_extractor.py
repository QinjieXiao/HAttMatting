import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


# print(model.conv1)
# summary(model,(3, 320, 320))

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = torch.hub.load('pytorch/vision', 'resnext50_32x4d', pretrained=True)
        self.conv1 = model.conv1
        self.forward_path = nn.Sequential(
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,   
            model.layer2,
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.forward_path(x)
        return F.log_softmax(x)


if __name__ == "__main__":
    model = FeatureExtractor()
    # model = torch.hub.load('pytorch/vision', 'resnext50_32x4d', pretrained=True)
    # print(model)
    summary(model, (3, 320, 320), depth=5)