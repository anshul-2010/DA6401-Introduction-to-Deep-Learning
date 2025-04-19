import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from dataloader import data_loader
from cnn_model import CNN_Model
from predict import predict

transform_valid = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

best_params = {
        'epoch': 50,
        'dropout': 0.3,
        'number_of_filters': 32,
        'filter_organisation': 2,
        'size_filter': 3,
        'data_augmentation': 'Yes',
        'batch_normalization': 'Yes',
        'learning_rate': 0.001,
        'optimiser_fn': 'adam',
        'activation_fn': 'relu',
        'dense': 256,
        'stride': 2    
    }

best_model = CNN_Model(
        num_filters=best_params["number_of_filters"],
        filter_organisation=best_params["filter_organisation"],
        size_filter=best_params["size_filter"],
        dropout=best_params["dropout"],
        activation_fn=best_params["activation_fn"],
        dense_units=best_params["dense"],
        stride=best_params["stride"],
        batch_normalization=best_params["batch_normalization"]
    )

# Set up optimizer
if best_params["optimiser_fn"] == 'adam':
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])
elif best_params["optimiser_fn"] == 'sgd':
    optimizer = optim.SGD(best_model.parameters(), lr=best_params["learning_rate"], momentum=0.0)
elif best_params["optimiser_fn"] == 'momentum':
    optimizer = optim.SGD(best_model.parameters(), lr=best_params["learning_rate"], momentum=0.9)

# Loss function
criterion = nn.CrossEntropyLoss()

# Load the dataloaders
data_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/DA6401/DA6401-Introduction-to-Deep-Learning/Assignment_2/inaturalist_12K"
train_path = os.path.join(data_path, "train")
train_loader, valid_loader = data_loader(data_path, 'train', batch_size=32, val_split=0.2)
test_path = os.path.join(data_path, "val")
test_loader, _ = data_loader(data_path, 'test', batch_size=32, val_split=0.2)

# Training loop
best_model.train()
for epoch in range(best_params["epoch"]):
    running_loss = 0.0
    running_val_loss = 0.0
    correct = 0
    correct_val = 0
    total = 0
    total_val = 0
    # Training phase
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct / total
    
    # Validation phase
    best_model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct_val / total_val
    
    print(f"Epoch [{epoch+1}/{epoch}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%", 
          f"Validation Loss: {running_val_loss/len(valid_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

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
        output = best_model(input_tensor)
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
    
predict(best_model)