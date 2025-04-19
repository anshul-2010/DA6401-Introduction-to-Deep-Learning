import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
img_width, img_height = 128, 128
batch_size = 32
n_filters = 32

data_path = "../Data/inaturalist_12K/test"
model_path = "../TrainedModel/Best_Model.pth"

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
test_dataset = ImageFolder(root=data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Class names
class_names = test_dataset.classes
class_to_idx = test_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model (e.g., ResNet18 as placeholder)
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Get the first conv layer
first_conv_layer = model.conv1
filters = first_conv_layer.weight.data.clone().cpu()

# Normalize filters
filter_min, filter_max = filters.min(), filters.max()
filters = (filters - filter_min) / (filter_max - filter_min)

# Plot filters
fig, axs = plt.subplots(3, n_filters, figsize=(20, 4))
for i in range(n_filters):
    for j in range(3):
        axs[j, i].imshow(filters[i][j], cmap='gray')
        axs[j, i].axis('off')
plt.tight_layout()
plt.show()

# Select a random image from the test set
images, labels = next(iter(test_loader))
img_index = np.random.randint(0, batch_size)
img = images[img_index:img_index+1].to(device)
label = labels[img_index].item()

# Plot input image
plt.figure()
plt.title(f"True label: {idx_to_class[label]}")
plt.imshow(np.transpose(images[img_index].numpy(), (1, 2, 0)))
plt.axis('off')
plt.show()

# Hook to get feature maps
def hook_fn(module, input, output):
    global feature_maps
    feature_maps = output.detach().cpu()

hook = first_conv_layer.register_forward_hook(hook_fn)

# Forward pass
_ = model(img)

# Plot feature maps
ROWS, COLUMNS = 4, 8
fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(12, 6))
for i in range(ROWS * COLUMNS):
    axs[i // COLUMNS, i % COLUMNS].imshow(feature_maps[0, i], cmap='gray')
    axs[i // COLUMNS, i % COLUMNS].axis('off')
plt.tight_layout()
plt.show()

# Clean up hook
hook.remove()
