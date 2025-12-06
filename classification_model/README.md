# Multi-Task Waste Recognition

Evaluating Deep Learning Models and Data Manipulation on Limited Real-World Trash Data

## Overview

This repository contains a comprehensive waste recognition system that performs multi-task learning on the TACO (Trash Annotations in Context) dataset. The project includes object classification, object detection, and instance segmentation tasks.

## Classification Task

### Introduction

The classification task focuses on categorizing waste items into 4 main classes: **plastic**, **glass**, **paper**, and **unsorted**. This is achieved by mapping the original 60 TACO dataset categories into these 4 broader classes using a custom `CLASS_MAP`.

### Key Features

- **State-of-the-art Architecture**: Uses ConvNeXt-Large, a modern CNN architecture that combines the best of CNNs and Transformers
- **Optimized Data Pipeline**: Pre-cropped images for maximum GPU utilization (90%+ vs 10-20% with on-the-fly cropping)
- **Class Imbalance Handling**: Implements weighted CrossEntropyLoss to address severe class imbalance in the dataset
- **Comprehensive Metrics**: Tracks macro F1, weighted F1, and per-class metrics (precision, recall, F1)
- **GPU Acceleration**: Optimized data loading with 8 workers and batch size 32

### Dataset

The model is trained on the **TACO (Trash Annotations in Context)** dataset, which contains:
- 1,500 images
- 4,784 annotations
- 60 original categories mapped to 4 classes

#### Class Distribution

The dataset shows significant class imbalance:
- **Plastic**: ~51% (majority class)
- **Unsorted**: ~31%
- **Paper**: ~12%
- **Glass**: ~5% (minority class)

### Model Architecture

**ConvNeXt-Large** with custom classifier:
- Base: ConvNeXt-Large (ImageNet pretrained)
- Classifier: 
  - LayerNorm
  - Dropout (0.3)
  - Linear (1536 → 512)
  - GELU activation
  - Dropout (0.2)
  - Linear (512 → 4 classes)
- Total parameters: ~197M

### Training Configuration

- **Epochs**: 25
- **Batch Size**: 32
- **Learning Rate**: 1e-4 (with cosine annealing)
- **Optimizer**: AdamW (weight decay: 1e-4)
- **Loss Function**: CrossEntropyLoss with inverse frequency class weights
- **Data Augmentation**: 
  - Random resized crop
  - Horizontal/Vertical flips
  - Rotation (±15°)
  - Color jitter
  - Random affine transformations
  - Random erasing

### Results

#### Validation Performance
- **Accuracy**: 78.94%
- **Macro F1**: 0.7406
- **Weighted F1**: 0.7885

#### Test Performance
- **Accuracy**: 77.39%
- **Macro F1**: 0.7458
- **Weighted F1**: 0.7746

#### Per-Class Performance (Test Set)

| Class     | F1 Score | Precision | Recall | Support |
|-----------|----------|-----------|--------|---------|
| Plastic   | 0.816    | 0.814     | 0.817  | 230     |
| Glass     | 0.765    | 0.765     | 0.765  | 17      |
| Unsorted  | 0.743    | 0.769     | 0.719  | 139     |
| Paper     | 0.660    | 0.608     | 0.721  | 43      |

### Usage

#### Prerequisites

```bash
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install tqdm
pip install scikit-learn
pip install numpy
```

#### Running the Notebook

1. **Setup**: Run the setup cell to import libraries and define `CLASS_MAP`
2. **Download Dataset**: Run the data import cell to download TACO dataset
3. **Pre-process Images**: Run the pre-processing cell to crop images (one-time operation)
4. **Train Model**: Run the training cells to train the ConvNeXt model
5. **Evaluate**: Run the evaluation cell to test on the test set

#### Loading the Trained Model

```python
import torch
from torchvision import models

# Load checkpoint
checkpoint = torch.load('convnext_taco_classification.pth', weights_only=False)

# Recreate model architecture
model = models.convnext_large(weights=None)
model.classifier = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.LayerNorm((1536,), eps=1e-6, elementwise_affine=True),
    nn.Dropout(0.3),
    nn.Linear(1536, 512),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(512, 4)
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get class names
class_names = checkpoint['category_names']  # ['glass', 'paper', 'plastic', 'unsorted']
```

### File Structure

```
.
├── classification_notebook.ipynb    # Main classification training notebook
├── TACO/                            # TACO dataset directory
│   ├── data/
│   │   ├── annotations.json         # Dataset annotations
│   │   └── batch_*/                 # Image batches
│   └── ...
└── taco_cropped_v1/                 # Pre-processed cropped images
    ├── train/
    │   ├── plastic/
    │   ├── glass/
    │   ├── paper/
    │   └── unsorted/
    ├── val/
    └── test/
```

### CLASS_MAP Details

The `CLASS_MAP` maps 60 TACO categories to 4 classes:

- **Plastic** (25 categories): All plastic items including bottles, containers, bags, etc.
- **Glass** (4 categories): Glass bottles, jars, cups, broken glass
- **Paper** (14 categories): Paper, cartons, tissues, cardboard, etc.
- **Unsorted** (17 categories): Metal items, food waste, composite materials, unknown items

### Key Optimizations

1. **Pre-cropping**: Images are pre-cropped using ground truth bounding boxes/masks, eliminating CPU bottleneck during training
2. **Class Weights**: Inverse frequency weighting helps the model learn from minority classes
3. **Data Augmentation**: Strong augmentation for minority classes, moderate for majority classes
4. **Efficient Data Loading**: 8 workers with persistent workers for faster data loading

### Future Improvements

- Experiment with different architectures (Vision Transformers, EfficientNet)
- Implement focal loss for better handling of hard examples
- Add more sophisticated data augmentation techniques
- Explore transfer learning from other waste classification datasets
- Implement ensemble methods

### Citation

If you use this code or dataset, please cite:

```bibtex
@article{taco2019,
  title={TACO: Trash Annotations in Context for Litter Detection},
  author={Pedro F. Proença and Pedro Simões},
  journal={arXiv preprint arXiv:2003.06975},
  year={2020}
}
```

### License

[Add your license information here]

### Contact

[Add your contact information here]
