import os
import torch
from torchvision import models, transforms
from PIL import Image

# Charger le modèle pré-entraîné
#model = models.resnet18(pretrained=True)
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

#Labels ImageNet (1000 classes)
#LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
#import urllib
#labels = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").split("\n")

with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Prétraitement de l’image (comme pour ImageNet)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, class_idx = torch.max(probabilities, dim=0)
        predicted_class = labels[class_idx]

    return predicted_class, round(confidence.item() * 100, 2)