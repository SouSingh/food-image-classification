import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim import Adam
from torch import nn, save, load
import json


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(64-6)*(64-6), 101)
        )

    def forward(self, x):
        return self.model(x)


def preprocess(img_path):
        clf = ImageClassifier()
        with open('model_state.pt', 'rb') as f:
            clf.load_state_dict(load(f))
        with open("class_name.txt", "r") as file:
            class_names_loaded = file.read().splitlines()
        img = Image.open(img_path)
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(64,64))])
        im = data_transform(img).unsqueeze(0)
        label = torch.argmax(clf(im))
        class_name = class_names_loaded[label]
        return class_name




