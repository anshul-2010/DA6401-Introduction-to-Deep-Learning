# CNN-Based Classification on iNaturalist 12K Dataset
This sub-repository contains a complete workflow for training, evaluating, and interpreting a CNN-based image classifier on the [iNaturalist 12K dataset](https://www.kaggle.com/competitions/inaturalist-2021-fgvc8/data). The implementation includes training, evaluation, filter visualization, guided backpropagation, and hyperparameter tuning using Weights & Biases sweeps.

## Project Structure

```
.
├── cnn_model.py
├── dataloader.py
├── evaluate.py
├── filter_visualization.py
├── guided_backpropagation.py
├── predict.py
├── run_sweep.py
├── sweep_config.py
├── train.py
├── PartA_Q2.ipynb
├── PartA_Q4.ipynb
└── README.md
```

---

## Setup Instructions

### 1. **Install Dependencies**

```bash
pip install torch torchvision matplotlib wandb numpy tqdm
```

If using Jupyter notebooks:

```bash
pip install notebook
```

Make sure to log in to Weights & Biases before running any script involving `wandb`:

```bash
wandb login
```

---

## Dataset Structure

Place the `inaturalist_12K` dataset in a folder named `../Data/inaturalist_12K/` relative to this repo. The structure should be:

```
inaturalist_12K/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

---

## Tasks and Files

### 1. `cnn_model.py`

Defines the CNN architecture used for training and evaluation.

- Modify this file if you want to try a different architecture or add dropout/batchnorm layers.

---

### 2. `dataloader.py`

Handles image loading, transformation, and batch generation using PyTorch `DataLoader`.

- Run this as part of the training/evaluation pipeline.
- Customize image resizing, augmentations, etc., here.

---

### 3. `train.py`

Main training script using the model and dataloader.

- Logs training and validation metrics to `wandb`.
- Saves the best model checkpoint based on validation accuracy.

---

### 4. `evaluate.py`

Evaluate a trained model on the test dataset.

- Prints accuracy, precision, recall, and F1-score.
- Optionally logs confusion matrix.

---

### 5. `filter_visualization.py`

Visualizes the learned filters of the first convolutional layer and the corresponding feature maps.

```bash
python filter_visualization.py --model_path ./TrainedModel/Best_Model.pth
```

- Displays both filters and their activations on a sample image.

---

### 6. `guided_backpropagation.py`

Implements guided backpropagation to visualize input regions responsible for specific class activations.

```bash
python guided_backpropagation.py --model_path ./TrainedModel/Best_Model.pth --target_class butterfly
```

- Generates a guided backprop gradient visualization.

---

### 7. `predict.py`

Make a prediction on a single image.

```bash
python predict.py --image_path sample.jpg --model_path ./TrainedModel/Best_Model.pth
```

- Outputs the predicted class label and class probabilities.

---

### 8. `run_sweep.py` & `sweep_config.py`

Used for hyperparameter tuning with Weights & Biases sweeps.

#### Step 1: Define sweep config in `sweep_config.py`
#### Step 2: Run a sweep

```bash
wandb sweep sweep_config.py
python run_sweep.py
```

## Notebooks

### `PartA_Q2.ipynb`

- Interactive notebook for training and inspecting model behavior.
- Useful for experimenting with small datasets or settings.

### `PartA_Q4.ipynb`

- Loads the best model, evaluates it on test data.
- Includes confusion matrix, misclassified samples, and visualization of feature importance.


## Example Training Command

```bash
python run_sweep.py
```

---

## Example Evaluation Command

```bash
python evaluate.py
python filter_visualization.py
python guided_backpropagation.py
```

## Notes

- Make sure to adjust GPU/CPU device settings in each file as needed.
- For large batch sizes, ensure enough VRAM is available or reduce image resolution.
- All visualizations are done using `matplotlib` and can be saved using `plt.savefig()` if desired.
- All the paths are defined with respect to the local environment. Make necessary changes to replicate for relative path
