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

## Average Loss per Epoch

```
Epoch [1/50] Loss: 1.9027
Epoch [2/50] Loss: 1.7469
Epoch [3/50] Loss: 1.6538
Epoch [4/50] Loss: 1.5664
Epoch [5/50] Loss: 1.4644
Epoch [6/50] Loss: 1.3520
Epoch [7/50] Loss: 1.2582
Epoch [8/50] Loss: 1.1780
Epoch [9/50] Loss: 1.0745
Epoch [10/50] Loss: 0.9533
Epoch [11/50] Loss: 0.8675
Epoch [12/50] Loss: 0.7649
Epoch [13/50] Loss: 0.7041
Epoch [14/50] Loss: 0.6342
Epoch [15/50] Loss: 0.5781
Epoch [16/50] Loss: 0.5154
Epoch [17/50] Loss: 0.4578
Epoch [18/50] Loss: 0.4208
Epoch [19/50] Loss: 0.4014
Epoch [20/50] Loss: 0.4039
Epoch [21/50] Loss: 0.3399
Epoch [22/50] Loss: 0.3027
Epoch [23/50] Loss: 0.2925
Epoch [24/50] Loss: 0.2557
Epoch [25/50] Loss: 0.2709
Epoch [26/50] Loss: 0.2371
Epoch [27/50] Loss: 0.2268
Epoch [28/50] Loss: 0.2206
Epoch [29/50] Loss: 0.2044
Epoch [30/50] Loss: 0.2029
Epoch [31/50] Loss: 0.1951
Epoch [32/50] Loss: 0.1737
Epoch [33/50] Loss: 0.1575
Epoch [34/50] Loss: 0.1756
Epoch [35/50] Loss: 0.1322
Epoch [36/50] Loss: 0.1635
Epoch [37/50] Loss: 0.1542
Epoch [38/50] Loss: 0.1191
Epoch [39/50] Loss: 0.1082
Epoch [40/50] Loss: 0.1195
Epoch [41/50] Loss: 0.1120
Epoch [42/50] Loss: 0.1152
Epoch [43/50] Loss: 0.1115
Epoch [44/50] Loss: 0.0920
Epoch [45/50] Loss: 0.1234
Epoch [46/50] Loss: 0.1182
Epoch [47/50] Loss: 0.1167
Epoch [48/50] Loss: 0.1037
Epoch [49/50] Loss: 0.1203
Epoch [50/50] Loss: 0.1006
```
## Results
| Metric     | Score |
|------------|-------|
| Accuracy   |    0.825   |
| Precision  |     0.805  |
| Recall     |   0.795    |
