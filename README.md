# Waste Object Detection – Project README

This repository contains a comprehensive workflow for developing waste detection and segmentation systems. It utilizes the TACO dataset to train **YOLOv8-based** models. The project is structured to guide the user from raw data acquisition and custom dataset creation through to advanced training techniques such as tiling and offline augmentation.

## Folder Structure

The repository is organized to separate model outputs from source code. The root directory contains the following structure:

```text
object detection/
│
├── v1_outputs/                # Output directory for Baseline model artifacts
├── v2_outputs/                # Output directory for YOLOv8s model artifacts
├── v3_outputs/                # Output directory for YOLOv8-P2 model artifacts
├── v4_outputs/                # Output directory for Tiling model artifacts
│
├── 0_taco_dataset_download.ipynb         # Data acquisition script
├── 2_custom_4cats.ipynb                  # Detection dataset preparation
├── 3_custom_4cats_seg.ipynb              # Segmentation dataset preparation
├── v1_yolo_model.ipynb                   # Baseline training script
├── v2_yolov8s_model.ipynb                # Standard YOLOv8 training script
├── v3_yolov8s_p2_model.ipynb             # Small-object optimized training
├── v4_yolov8s_tiling.ipynb               # Tiling strategy training
├── v5_yolov8s_offline_augmentation.ipynb # Data augmentation pipeline
```

-----

## Step 1: Dataset Preparation Pipeline

Before initiating any training loops, the data must be downloaded, filtered, and formatted. Execute the following notebooks in order:

### 1\. Data Acquisition

**File:** `0_taco_dataset_download.ipynb`

This notebook establishes the project foundation by downloading the **TACO (Trash Annotations in Context)** dataset. It handles the extraction and organization of raw images and annotation files into a structured working directory required for subsequent processing steps.

### 2\. Detection Dataset Configuration

**File:** `2_custom_4cats.ipynb`

This script processes the raw TACO annotations to create a custom dataset specifically for object detection. It filters the complex TACO class ontology into a consolidated **four-category** schema. This step ensures the model focuses on the specific waste categories relevant to this project.

### 3\. Segmentation Mask Generation

**File:** `3_custom_4cats_seg.ipynb`

To enable instance segmentation tasks, this notebook generates precise segmentation masks for the four-category dataset created in the previous step. Running this is essential if you intend to train models that require polygon masks rather than simple bounding boxes.

-----

## Step 2: Model Training Strategies

The repository offers multiple training configurations, ranging from baselines to specialized architectures for small object detection. 

Here is a detailed explanation of each model version found in the repository. The project progresses from a simple baseline to specialized architectures designed to solve specific problems like detecting small trash items or handling high-resolution images.

### `v1_yolo_model.ipynb` – The Baseline
* **Purpose:** This model serves as the initial benchmark or "sanity check" for the project.
* **How it works:** It likely uses a standard, out-of-the-box YOLO configuration without extensive customization.
* **Why use it:** Before implementing complex optimizations, you need a baseline to compare against. If later models (like v3 or v4) do not perform better than this one, it indicates that the advanced techniques are not providing value. It verifies that the dataset is formatted correctly and that the training pipeline is functional.

### `v2_yolov8s_model.ipynb` – Standard YOLOv8 Small
* **Purpose:** This is the primary "workhorse" model for general object detection.
* **How it works:** It utilizes the **YOLOv8s (Small)** architecture. The "s" stands for small, indicating it has fewer parameters than the Medium (m), Large (l), or Extra Large (x) versions.

### `v3_yolov8s_p2_model.ipynb` – Small Object Optimization (P2 Layer)
* **Purpose:** This model is specifically engineered to detect **small objects**, which is a common challenge in waste detection (e.g., cigarette butts, bottle caps).
* **How it works:** Standard YOLO models typically downsample the image significantly (often by a factor of 32 at the deepest layer), causing small objects to disappear from the feature maps. The **P2** architecture adds an extra detection head that operates at a higher resolution (downsampled only by a factor of 4 or 8).


### `v4_yolov8s_tiling.ipynb` – High-Resolution Tiling
* **Purpose:** This approach handles **high-resolution images** or wide scenes where resizing the entire image to the model's input size (e.g., 640x640) would destroy image details.
* **How it works:** Instead of resizing the whole image at once, the image is sliced into smaller distinct patches (tiles). The model runs detection on each tile independently.


### `v5_yolov8s_offline_augmentation.ipynb` – Dataset Expansion
* **Purpose:** While not a model training script itself, this version represents a change in the *data* strategy rather than the model architecture.
* **How it works:** It applies offline transformations (rotations, color shifts, flips) to the images *before* training begins, permanently increasing the size of the dataset.

-----

## Technical Notes & Requirements

  * **Dependencies:** Ensure a Python environment is active with **PyTorch** and **Ultralytics** installed.
  * **Path Configuration:** Review the top cells of each notebook to ensure dataset paths match your local file system structure.
  * **Task Support:** This workflow allows for switching between **Object Detection** (bounding boxes) and **Instance Segmentation** (masks) by selecting the appropriate dataset preparation script (Step 1) and model configuration (Step 2).
