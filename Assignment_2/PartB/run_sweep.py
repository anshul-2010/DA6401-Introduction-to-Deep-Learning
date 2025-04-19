import wandb
from train import train_model
from sweep_config import sweep_configuration
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from torch.optim.lr_scheduler import StepLR
from models import get_model

wandb.login()

# Sweep function
batch_size = 32
train_path = 'C:\\Users\\Dell\\Desktop\\Courses\\Sem_VIII\\DA6401\\DA6401-Introduction-to-Deep-Learning\\Assignment_2\\inaturalist_12K\\train'

def main():
    wandb.init(project="DA6401-Assignment-2")
    config = wandb.config
    num_classes = 10

    model = get_model(config.base, num_classes)
    
    # Freeze all parameters first
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Then selectively unfreeze classifier layers
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            param.requires_grad = True
            
    # Data Preparation with PyTorch DataLoader
    if config.base == 'inception' or config.base == 'inceptionresnet':
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

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
    valid_dataset = datasets.ImageFolder(root=train_path, transform=transform_valid)

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config.optimizer_fn)(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    print("Training frozen base...")
    train_model(model, [train_loader, valid_loader], criterion, optimizer, 5)

    if config.ft_bool == 'Yes':
        print("Fine-tuning model...")
        # Unfreeze last few layers
        child_counter = 0
        for child in model.children():
            child_counter += 1
            if child_counter > 5:  # unfreeze last blocks
                for param in child.parameters():
                    param.requires_grad = True

        optimizer = getattr(optim, config.optimizer_fn)(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate / 10)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

        train_model(model, [train_loader, valid_loader], criterion, optimizer, 10, scheduler=scheduler)

sweep_config = sweep_configuration()

sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment-2", entity="anshul_2010-indian-institute-of-technology-madras")

# Run the sweep agent
# This will run the train function with different hyperparameters 15 times
wandb.agent(sweep_id, function=main, count=15)