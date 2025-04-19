# Transfer Learning with Pretrained CNN-Based Classification on iNaturalist 12K Dataset
This sub-repository contains a complete workflow for training, evaluating, and interpreting a CNN-based image classifier on the [iNaturalist 12K dataset](https://www.kaggle.com/competitions/inaturalist-2021-fgvc8/data) using multiple pretrained CNN architectures (ResNet, MobileNet, Inception, etc.) with transfer learning and fine-tuning. It includes a hyperparameter sweep using Weights & Biases (W&B) to find optimal configurations.
## Project Structure

```
.
├── dataloader.py
├── evaluate.py
├── models.py
├── run_sweep.py
├── sweep_config.py
├── train.py
├── PartB_Q2.ipynb
└── README.md
```

---

## Setup Instructions

### 1. **Install Dependencies**

```bash
pip install torch torchvision matplotlib wandb numpy tqdm timm
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

### 1. `dataloader.py`

Handles image loading, transformation, and batch generation using PyTorch `DataLoader`.

- Run this as part of the training/evaluation pipeline.
- Customize image resizing, augmentations, etc., here.

---

### 2. `models.py`

Defines the pretrained CNN architecture imports used for training and evaluation.

- Modify this file if you want to try a different architecture or add dropout/batchnorm layers.
- It includes the following models
  - ResNet
  - MobileNet
  - Inception
  - InceptionResNet

---

### 3. `train.py`

Main training script using the model and dataloader.

- Logs training and validation metrics to `wandb`.
- Saves the best model checkpoint based on validation accuracy.

---

### 4. `evaluate.py`

Evaluate a trained model on the test dataset.

- Prints accuracy on the trained as well as test dataset.
- Optionally logs confusion matrix.

---


### 5. `run_sweep.py` & `sweep_config.py`

Used for hyperparameter tuning with Weights & Biases sweeps.

#### Step 1: Define sweep config in `sweep_config.py`
#### Step 2: Run a sweep

```bash
wandb sweep sweep_config.py
python run_sweep.py
```

## Notebooks

### `PartB_Q2.ipynb`

- Interactive notebook for training and inspecting model behavior.
- Useful for experimenting with small datasets or settings.

## Example Training Command

```bash
python run_sweep.py
```

---

## Example Evaluation Command

```bash
python evaluate.py
```

## Notes

- Make sure to adjust GPU/CPU device settings in each file as needed.
- For large batch sizes, ensure enough VRAM is available or reduce image resolution.
- All visualizations are done using `matplotlib` and can be saved using `plt.savefig()` if desired.
- All the paths are defined with respect to the local environment. Make necessary changes to replicate for relative path
