# Waste Object Detection – Project README

This repository provides a complete workflow for preparing a custom waste-detection dataset and training YOLOv8-based object detection and segmentation models. It includes dataset preparation scripts, augmentation pipelines, and multiple model-training configurations.

---

## Folder Structure

```
object detection/
│
├── v1_outputs/
├── v2_outputs/
├── v3_outputs/
├── v4_outputs/
│   ├── 0_taco_dataset_download.ipynb
│   ├── 2_custom_4cats.ipynb
│   ├── 3_custom_4cats_seg.ipynb
│   ├── v1_yolo_model.ipynb
│   ├── v2_yolov8s_model.ipynb
│   ├── v3_yolov8s_p2_model.ipynb
│   ├── v4_yolov8s_tiling.ipynb
│   ├── v5_yolov8s_offline_augmentation.ipynb
```

---

## Step 1: Dataset Preparation

Before training any detection or segmentation models, run the dataset preparation notebooks in the following order:

### 1. `0_taco_dataset_download.ipynb`

Downloads the TACO dataset and organizes the raw images and annotations into a working directory.

### 2. `2_custom_4cats.ipynb`

Creates a custom four-category waste dataset suitable for object detection.
This notebook handles

### 3. `3_custom_4cats_seg.ipynb`

Generates segmentation masks for the four-category dataset.
These masks allow training of segmentation-capable YOLO models.

After completing these notebooks, the dataset will be fully prepared for use in the training pipelines.

---

## Step 2: Model Training

Once the dataset is ready, choose the appropriate training notebook:

### `v1_yolo_model.ipynb`

Baseline YOLO model training.

### `v2_yolov8s_model.ipynb`

Training a YOLOv8-small object detection model.

### `v3_yolov8s_p2_model.ipynb`

Training a YOLOv8-P2 model optimized for small object detection.

### `v4_yolov8s_tiling.ipynb`

Tiling-based training for high-resolution images or scenes with many small items.

### `v5_yolov8s_offline_augmentation.ipynb`

Offline dataset augmentation pipeline used to expand dataset variety before training.

---

## Notes
* Ensure that required dependencies (PyTorch, Ultralytics) are installed before running the notebooks.
* Adjust dataset paths inside each notebook to match your local environment.
* This workflow supports both detection and segmentation tasks depending on the chosen model and dataset configuration.

---
