# Car Brand Classifier
 
A convolutional neural network (CNN) built with PyTorch that classifies car images into 7 brand/model categories from the following Kaggle dataset: https://www.kaggle.com/datasets/kshitij192/cars-image-dataset.
 
## Classes
 
The model distinguishes between the following 7 classes:
 
- Audi
- Hyundai Creta
- Mahindra Scorpio
- Rolls Royce
- Swift
- Tata Safari
- Toyota Innova
## Model Architecture
 
`CarClassifier` is a custom CNN consisting of:
 
- 4 convolutional blocks (`Conv2d` → `ReLU` → `MaxPool2d`), with channel depths 3 → 32 → 64 → 128 → 256
- A fully connected head: `Linear(256*14*14, 512)` → `ReLU` → `Dropout(0.35)` → `Linear(512, 7)`
Input images are resized to **224×224**, so after 4 pooling layers (each halving spatial dimensions) the feature map is 14×14 before flattening.
 
## Dataset
 
The dataset is expected to follow the `torchvision.datasets.ImageFolder` structure, with one subdirectory per class:
 
```
car_classifier/
├── training/
│   ├── Audi/
│   ├── Hyundai Creta/
│   ├── Mahindra Scorpio/
│   ├── Rolls Royce/
│   ├── Swift/
│   ├── Tata Safari/
│   └── Toyota Innova/
└── test/
    ├── Audi/
    ├── Hyundai Creta/
    ├── ...
```
 
> **Note:** `ImageFolder` assigns class labels automatically based on the alphabetical order of subdirectories — it does **not** use any hardcoded class name list. Make sure both `training/` and `test/` contain exactly the same 7 class folders (no stray/hidden folders), or training will fail with a CUDA `device-side assert` error due to out-of-range labels.
 
### Data Augmentation
 
Training images are augmented with:
- Random horizontal flip
- Random rotation (±15°)
- Random color jitter (brightness, contrast, hue)
Both training and test images are normalized with mean/std `(0.5, 0.5, 0.5)`.
 
## Requirements
 
```
torch
torchvision
torchmetrics
numpy
matplotlib
```
 
Install with:
 
```bash
pip install torch torchvision torchmetrics numpy matplotlib
```
 
A CUDA-capable GPU is recommended but not required (the script automatically falls back to CPU).
 
## Usage
 
1. Update the dataset paths in `car_classifier.py` to point to your local `training/` and `test/` directories:
```python
training_data = datasets.ImageFolder(r'path\to\training', transform=train_transforms)
test_data = datasets.ImageFolder(r'path\to\test', transform=test_transforms)
```

 
The script will:
- Display a sample batch of training images
- Train the model for 50 epochs (Adam optimizer, `lr=0.001`, cross-entropy loss)
- Evaluate on the test set, reporting **Accuracy**, **Precision** (macro-averaged), and **Recall** (macro-averaged)
## Training Configuration
 
| Parameter        | Value              |
|-------------------|--------------------|
| Batch size        | 64                 |
| Epochs             | 50                 |
| Optimizer          | Adam               |
| Learning rate      | 0.001              |
| Loss function      | CrossEntropyLoss   |
| Dropout            | 0.35               |
 
