from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

def predict(best_model, test_dir, class_names, img_height=224, img_width=224, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Set the model to evaluation mode
    best_model.eval()
    best_model.to(device)

    # Define the transform for preprocessing images
    preprocess = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    plt.figure(figsize=(25, 25))
    id = 0

    for subdir in os.listdir(test_dir):
        files = [random.choice(os.listdir(os.path.join(test_dir, subdir))) for _ in range(3)]
        index = 1

        for file in files:
            path = os.path.join(test_dir, subdir, file)

            # Load and preprocess the image
            image = Image.open(path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = best_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            predicted_class = class_names[probabilities.argmax().item()]
            confidence = probabilities.max().item()

            # Plot
            plt.subplot(10, 10, id * 10 + index)
            plt.imshow(image)
            plt.title(f"{predicted_class}, {confidence * 100:.2f}%")
            plt.axis('off')

            index += 1
        id += 1

    plt.tight_layout()
    plt.show()