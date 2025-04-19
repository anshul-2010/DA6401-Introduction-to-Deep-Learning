import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from cnn_model import CNN_Model
from dataloader import data_loader

def train_model(epoch, dropout, number_of_filters, filter_organisation, size_filter, data_augmentation, batch_normalization, learning_rate, optimizer_fn, activation_fn, dense, stride):
    # Model setup
    model = CNN_Model(
        num_filters=number_of_filters,
        filter_organisation=filter_organisation,
        size_filter=size_filter,
        dropout=dropout,
        activation_fn=activation_fn,
        dense_units=dense,
        stride=stride,
        batch_normalization=batch_normalization
    )
    
    # Set up optimizer
    if optimizer_fn == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_fn == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    elif optimizer_fn == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Load the dataloaders
    data_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/DA6401/DA6401-Introduction-to-Deep-Learning/Assignment_2/inaturalist_12K"
    train_path = os.path.join(data_path, "train")
    train_loader, valid_loader = data_loader(data_path, 'train', batch_size=32, val_split=0.2)
    
    # Training loop
    model.train()
    for epoch in range(epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epoch}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        
        wandb.log({
            'accuracy': train_accuracy,
            'loss': running_loss / len(train_loader)
        })
    
    wandb.log({'model_accuracy': train_accuracy})

    return model