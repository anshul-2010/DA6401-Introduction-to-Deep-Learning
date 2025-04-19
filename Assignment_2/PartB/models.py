# Define the CNN Model in PyTorch
import torch
import torch.nn as nn
from torchvision import models
import timm

# Load pretrained model
def get_model(base_name, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if base_name == 'resnet':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    elif base_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    elif base_name == 'inception':
        model = models.inception_v3(pretrained=True, aux_logits=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    elif base_name == 'inceptionresnet':
        model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
    
    return model.to(device)