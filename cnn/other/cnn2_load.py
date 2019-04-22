import torch
import torch.nn as nn
from torchvision import models


def main():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('cnn.pt', map_location='cpu'))
    model.eval()

if __name__ == '__main__':
    main()
