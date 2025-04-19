# DA7450 - Assignment 2 [BE21B005]

This assignment explores image classification and object detection using different deep learning techniques on the iNaturalist dataset and beyond.

## Part A: Convolutional Neural Network (CNN) Implementation from Scratch
---
This section details the implementation of a Convolutional Neural Network (CNN) from the ground up using a subset of the iNaturalist dataset. The code provides flexibility in experimenting with various network configurations and training parameters.

**Implemented Functionalities:**

* **Optimizers:** Stochastic Gradient Descent (SGD), Momentum, Adam
* **Activation Functions:** ReLU, Tanh
* **Initialization:** Random
* **Number of Filters:** Configurable as 16 or 32
* **Stride Values:** Configurable as 1 or 2

**Hyperparameters Explored:**

* **Dropout:** A regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 during training.
* **Batch Normalization:** A technique to accelerate training and improve stability by normalizing the input of each layer. Applied after the activation function of the output layer or before the activation function of the input layer.
* **Filter Organization:** Controls the number of filters in each convolutional layer of the CNN.
* **Data Augmentation:** Techniques used to artificially expand the training dataset by applying transformations to existing images, improving the model's generalization.

The final dense layer utilizes the **softmax** activation function for multi-class classification.

**Experimentation and Hyperparameter Tuning:**

The implementation is integrated with Weights & Biases (wandb) for effective experiment tracking and hyperparameter tuning. The following hyperparameter space is explored using wandb sweeps:

```yaml
epoch: [5, 7, 8, 10]
dropout: [0.1, 0.2, 0.3]
number_of_filters: [16, 32]
filter_organisation: [0.5, 1, 2]
size_filter: [3, 4]
data_augmentation: ['Yes', 'No']
batch normalization: ['Yes', 'No']
learning_rate: [0.001, 0.0001]
optimizer_fn: ['adam', 'sgd', 'momentum']
activation_fn: ['relu', 'tanh']
dense: [128, 256, 512]
stride: [1, 2]
```

**Related Files:**

* `PartA/PartA_Q2.ipynb`: Implements the CNN from scratch and performs hyperparameter search using wandb sweeps to identify the best configuration.
* `PartA/PartA_Q4.ipynb`: Visualizes the features learned by the best performing model using techniques like Guided Backpropagation, Grad-CAM, and Guided Grad-CAM.
* Other `.py` files are also there that orchestrate this process via scripts. Detailed instructions are there in each Task folder specifically.

## Part B: Fine-Tuning Pre-trained Models
---
This section focuses on leveraging the knowledge learned by models pre-trained on the large-scale ImageNet dataset and adapting them for the iNaturalist dataset through fine-tuning. The choice of ImageNet is motivated by its visual similarity to the iNaturalist dataset.

**Pre-trained Models Used:**

* Inception
* ResNet
* InceptionResNet
* Xception

**Training Functionalities:**

* **Optimizers:** Adam, Stochastic Gradient Descent (SGD)
* **Activation Functions:** ReLU, Tanh

**Key Hyperparameters for Fine-Tuning:**

* `K_ft`: Specifies the layer number from which the convolutional layers are unfrozen for fine-tuning.
* `Ft_bool`: A boolean flag (set to 'Yes' for fine-tuning) indicating whether to unfreeze and fine-tune layers.
* `In_epochs`: The number of training epochs to run with the pre-trained model (with most layers frozen).
* `Ft_epoch`: The number of training epochs to run after unfreezing specified layers for fine-tuning.
* `preprocess_input`: A function used to preprocess the input data according to the specific requirements of the pre-trained model.
* **Global Average Pooling:** Applied before the final classification layer to reduce the dimensionality of the feature maps and prepare the model for classification.

Similar to Part A, the final dense layer uses the **softmax** activation function.

**Experimentation and Hyperparameter Tuning:**

The fine-tuning process is also integrated with wandb for tracking and hyperparameter optimization. The following hyperparameter space is explored:

```yaml
base: ['inception', 'resnet', 'inceptionresnet', 'xception']
in_epoch: [4, 5, 10]
ft_epoch: [4, 5, 10]
ft_bool: ['Yes']
dropout: [0.1, 0.2]
optimizer_fn: ['adam', 'sgd']
activation_fn: ['relu', 'tanh']
dense: [256, 512]
```

**Related Files:**

* `PartB/PartB_Q2.iypnb`: Implements wandb sweeps to find the optimal hyperparameters for fine-tuning the pre-trained models and performs the fine-tuning process based on these parameters.
* Other `.py` files are there that orchestrate the entire process. Specific details in the folder.

## Part C: Object Detection using YOLOv3
---
This section explores object detection using the pre-trained YOLOv3 (You Only Look Once version 3) model.

**Implemented Functionalities:**

* Object detection in static image frames from videos.
* Object detection in video streams.
* Entendable to Real-time human detection using YOLOv3 and OpenCV.

**Related Files:**

* `Yolo_Imageai_Video.iypnb`: Implements object detection on images using the YOLOv3 model.
* `traffic.mp4`: Input video which was used for object detection.
* The output video with detected boundaries was too big to be uploaded on github.
