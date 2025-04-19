import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
IMG_SIZE = (128, 128)
DATAPATH = "../Data/inaturalist_12K/test"
BATCH_SIZE = 32
NUM_SAMPLE_IMAGES = 10

# Image transforms
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# Dataset and DataLoader
test_dataset = datasets.ImageFolder(root=DATAPATH, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = test_dataset.classes

# Load model (replace with your model and path)
model = torch.load("../TrainedModel/Best_Model")
model.to(device)
model.eval()

# Guided ReLU
class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        ctx.save_for_backward(input, positive_mask)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, positive_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        grad_input[grad_output <= 0] = 0
        return grad_input

# Replace ReLU with GuidedReLU
def replace_relu_with_guided_relu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedReLUWrapper())
        else:
            replace_relu_with_guided_relu(module)

class GuidedReLUWrapper(nn.Module):
    def forward(self, x):
        return GuidedReLU.apply(x)

# Hook to capture gradients
class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        replace_relu_with_guided_relu(self.model)

    def hook_layers(self, target_layer):
        def forward_hook(module, input, output):
            self.activation = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for name, module in self.model.named_modules():
            if name == target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_gradients(self, input_image, target_class):
        input_image = input_image.unsqueeze(0).to(device)
        input_image.requires_grad = True

        output = self.model(input_image)
        self.model.zero_grad()

        target = torch.zeros_like(output)
        target[0][target_class] = 1
        output.backward(gradient=target)

        return self.gradients[0].cpu().detach().numpy()

# Deprocess image for visualization
def deprocess_image(img):
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.25
    img += 0.5
    img = np.clip(img, 0, 1)
    img *= 255
    img = np.transpose(img, (1, 2, 0))
    return img.astype(np.uint8)

# Visualize
sample_batch = next(iter(test_loader))
images, labels = sample_batch

guided_bp = GuidedBackprop(model)
guided_bp.hook_layers("features.28")  # Replace with correct target layer name

fig, axs = plt.subplots(2, NUM_SAMPLE_IMAGES, figsize=(20, 5))

for i in range(NUM_SAMPLE_IMAGES):
    img = images[i]
    label = labels[i].item()

    gradients = guided_bp.generate_gradients(img, label)

    axs[0, i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
    axs[0, i].set_title(f"Label: {class_names[label]}")
    axs[0, i].axis('off')

    axs[1, i].imshow(deprocess_image(gradients))
    axs[1, i].set_title("Guided Grad")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()
