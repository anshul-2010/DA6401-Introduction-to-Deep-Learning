from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import StepLR
from models import get_model
from train import train_model

transform_valid = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

best_params = {
        'base': 'inceptionresnet',
        'in_epoch': 10,
        'ft_epoch': 6,
        'ft_bool': 'Yes',
        'dropout': 0.2,
        'learning_rate': 0.001,
        'optimizer_fn': 'Adam',
        'activation_fn': 'relu',
        'dense': 256
    }

num_classes = 10
model = get_model(best_params["base"], num_classes)

# Freeze all parameters first
for name, param in model.named_parameters():
    param.requires_grad = False

# Then selectively unfreeze classifier layers
for name, param in model.named_parameters():
    if 'fc' in name or 'classifier' in name:
        param.requires_grad = True
            
# Data Preparation with PyTorch DataLoader
if best_params["base"] == 'inception' or best_params["base"] == 'inceptionresnet':
    image_size = (299, 299)  # Inception and InceptionResNet require 299x299
else:
    image_size = (224, 224)  # Default size for other models

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_valid = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/DA6401/DA6401-Introduction-to-Deep-Learning/Assignment_2/inaturalist_12K/train"
test_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/DA6401/DA6401-Introduction-to-Deep-Learning/Assignment_2/inaturalist_12K/val"

train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
valid_dataset = datasets.ImageFolder(root=train_path, transform=transform_valid)

train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = getattr(optim, best_params["optimizer_fn"])(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params["learning_rate"])

# Loss function
criterion = nn.CrossEntropyLoss()

# Load the dataloaders
test_dataset = datasets.ImageFolder(root=test_path, transform=transform_valid)
extra_dataset = datasets.ImageFolder(root=test_path, transform=transform_valid)

test_size = int(0.8 * len(test_dataset))
xtra_size = len(test_dataset) - test_size
test_subset, xtra_subset = random_split(test_dataset, [test_size, xtra_size])

test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
xtra_loader = DataLoader(xtra_subset, batch_size=32, shuffle=False)

print("Training frozen base...")
train_model(model, [train_loader, valid_loader], criterion, optimizer, best_params["in_epoch"])

if best_params["ft_bool"] == 'Yes':
    print("Fine-tuning model...")
    # Unfreeze last few layers
    child_counter = 0
    for child in model.children():
        child_counter += 1
        if child_counter > 5:  # unfreeze last blocks
            for param in child.parameters():
                param.requires_grad = True

    optimizer = getattr(optim, best_params["optimizer_fn"])(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params["learning_rate"] / 10)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, [train_loader, valid_loader], criterion, optimizer, 10, scheduler=scheduler)

# Example testing  
test_path = 'C:\\Users\\Dell\\Desktop\\Courses\\Sem_VIII\\DA6401\\DA6401-Introduction-to-Deep-Learning\\Assignment_2\\inaturalist_12K\\val'
# Predict on test set
for i in range(1):
    # Randomly select a class
    random_class = random.choice(os.listdir(test_path))
    # Randomly select an image from that class
    random_image = random.choice(os.listdir(os.path.join(test_path, random_class)))
    # Construct the full path to the image
    image_path = os.path.join(test_path, random_class, random_image)
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Preprocess the image
    input_tensor = transform_valid(image).unsqueeze(0)  # Add batch dimension
    # Move the tensor to the same device as the model
    input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        # For InceptionV3, outputs are in InceptionOutputs, so we need to get the logits
        if isinstance(outputs, tuple):  # Check if it's an InceptionOutputs tuple
            outputs = outputs[0]  # Get the logits from the tuple
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get the predicted class and confidence
    class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 
                   'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']
    predicted_class = class_names[probabilities.argmax().item()]
    confidence = probabilities.max().item()
    # Print the result
    print(f"Predicted class: {predicted_class}, Confidence: {confidence * 100:.2f}%")
    # Display the image
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence * 100:.2f}%")
    plt.axis('off')
    plt.show()