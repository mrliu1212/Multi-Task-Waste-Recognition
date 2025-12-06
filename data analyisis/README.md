# Data Analysis – README

This directory contains all preprocessing and exploratory-analysis notebooks used to turn the raw TACO dataset into several YOLO-compatible datasets:

* a full detection dataset in YOLO format (`taco_yolo/`)
* a reduced four-class detection dataset (`yolo_4cats/`)
* a four-class segmentation dataset (`dataset_yolo_4cats_seg/`)

The notebooks also perform quantitative analysis on each version of the dataset to guide design choices (category grouping, split ratios, object-size thresholds, etc.).

The typical workflow is:

1. `0_taco_dataset_download.ipynb`
2. `1_create_yolo_dataset.ipynb`
3. `1_dataset_analysis.ipynb`
4. `2_custom_4cats.ipynb`
5. `2_dataset_analysis 4cats.ipynb`
6. `3_custom_4cats_seg.ipynb`

You do not have to run the analysis notebooks every time, but they document and verify the dataset design.

---

## 0_taco_dataset_download.ipynb

**Goal:** Obtain the official TACO dataset and verify that all images were downloaded correctly.

Main steps:

1. Clone the official TACO GitHub repository.
2. Install dependencies from `requirements.txt` and the COCO API.
3. Run the official `download.py` script from TACO.
4. Print the number of images per `batch_*` folder and the total number of images using a small helper:

   * `count_images(folder)` recursively scans for `jpg/jpeg/png` files.
   * Outputs per-batch counts and a final `TOTAL: X images`.

**Inputs**

* None (starts from an empty workspace).

**Outputs**

* `TACO/` repository.
* `TACO/data/` with:

  * COCO-style annotations (e.g., `annotations.json`).
  * Downloaded image batches (`batch_1`, `batch_2`, …).

This notebook is meant to be run once at the very beginning.

---

## 1_create_yolo_dataset.ipynb

**Goal:** Convert the original TACO COCO annotations into a full YOLO detection dataset (`taco_yolo/`), and create a standard Ultralytics-style `taco.yaml`.

Key configuration:

```python
COCO_JSON   = "TACO/data/annotations.json"
IMAGES_ROOT = "TACO/data"
YOLO_ROOT   = "taco_yolo"

TRAIN_PCT = 0.7
VAL_PCT   = 0.15       # remaining images become test set
SEED      = 42

USE_FINE_GRAINED_CLASSES = True  # TACO fine-grained vs grouped classes
```

Main steps:

1. **Load COCO annotations**

   * Read images, categories, and annotations from `annotations.json`.
   * Optionally decide whether to use the original fine-grained TACO categories or merge them into supercategories (`USE_FINE_GRAINED_CLASSES` flag).

2. **COCO-to-YOLO conversion**

   * For each annotation:

     * Convert COCO bounding boxes `(x, y, w, h)` into normalized YOLO format `(class_id, x_center, y_center, width, height)` in relative image coordinates.
   * Write one `.txt` label file per image under `taco_yolo/labels/`.

3. **Train/val/test split**

   * Split images into train/val/test according to `TRAIN_PCT` and `VAL_PCT`.
   * Use a fixed `SEED` for reproducibility.
   * Copy images into:

     * `taco_yolo/images/train`
     * `taco_yolo/images/val`
     * `taco_yolo/images/test`
   * Copy the corresponding label files into matching `labels/` subfolders.

4. **Generate the dataset config (`taco.yaml`)**

   * Create `taco_yolo/taco.yaml` with:

     * `path`: absolute path to `taco_yolo`
     * `train`: `images/train`
     * `val`: `images/val`
     * `test`: `images/test` (if used)
     * `names`: ordered list of class names.

5. **Sanity-check visualizations**

   * Utility to randomly pick some images, draw their bounding boxes with OpenCV, and display them with matplotlib.
   * Useful to confirm that coordinates and class IDs were converted correctly.

**Outputs**

* Directory `taco_yolo/` with:

  * `images/train`, `images/val`, `images/test`
  * `labels/train`, `labels/val`, `labels/test`
  * `taco.yaml`

---

## 1_dataset_analysis.ipynb

**Goal:** Analyze the full YOLO dataset (`taco_yolo`) to understand class imbalance, object sizes, image resolutions, and per-image instance counts.

Configuration and loading:

* Loads `taco_yolo/taco.yaml` to get class names.
* Uses glob to collect image paths for train/val/test.
* Provides helper functions such as:

  * `collect_images(path)`
  * `get_label_path(image_path)`

Main analyses:

1. **Class distribution**

   * Read every label file in `labels/train`, `labels/val`, and `labels/test`.
   * Count occurrences of each class ID.
   * Build a `pandas.DataFrame` with:

     * `class_id`, `class_name`, `instances`, and relative frequencies.
   * Print or display the top classes and overall distribution.

2. **Object size distribution**

   * For each bounding box, compute:

     * relative area: `(w * h)` in normalized coordinates (`area_ratio`).
   * Apply thresholds:

     ```python
     SMALL_THR  = 0.01  # area_ratio < 0.01  => "small"
     MEDIUM_THR = 0.05  # area_ratio < 0.05 => "medium", else "large"
     ```
   * Produce a table assigning each box to a size bucket (`small`, `medium`, `large`) and summarize how many boxes fall into each bucket.

3. **Image resolution and aspect ratio**

   * Use `PIL.Image.open` to inspect width and height of every image.
   * For each image, compute:

     * aspect ratio (`width / height`)
     * orientation (`portrait`, `landscape`, `square`)
   * Summarize:

     * basic statistics (min/mean/max width and height),
     * orientation counts,
     * the most common resolutions.

4. **Per-image instance counts**

   * Count how many bounding boxes appear in each training image.
   * Identify very dense images with many objects, which can be useful for debug or for special tiling strategies later.
   * Optionally compute cumulative curves showing what fraction of training images have at most N objects.

This notebook mainly produces tables and plots interactively; CSV export lines are present but commented out so you can enable them if you want persistent reports.

---

## 2_custom_4cats.ipynb

**Goal:** Build a more balanced detection dataset with four aggregated waste classes and no explicit test set, to maximize training data.

Key ideas (from notebook text):

* TACO has many fine-grained categories, many of which have very few samples.
* Based on the previous analysis, categories are regrouped into four higher-level classes so that each has enough training examples.
* The code also removes the test split and redistributes all data into train/val only, since this is not a benchmark competition but a model-development project.

Configuration:

```python
COCO_JSON   = "TACO/data/annotations.json"
IMAGES_ROOT = "TACO/data"
YOLO_ROOT   = "yolo_4cats"      # new detection dataset root

TRAIN_PCT = 0.85
VAL_PCT   = 0.15
SEED      = 42
```

Main steps:

1. **Define four meta-classes**

   * Map the original TACO category IDs into four main groups
     (e.g. Plastic, Paper, Glass, Unsorted; see segmentation notebook for the same design).
   * Any category not assigned to Plastic/Paper/Glass is mapped into `Unsorted`.

2. **Filter and regroup annotations**

   * Read COCO annotations.
   * Drop categories that are not used.
   * For each remaining annotation, convert its original category ID into the corresponding one of the four meta-classes.
   * Convert bounding boxes into YOLO format.

3. **Train/val split without a test set**

   * Split images into train and validation only (`TRAIN_PCT = 0.85`, `VAL_PCT = 0.15`).
   * Copy images and corresponding labels into:

     * `yolo_4cats/images/train`, `yolo_4cats/images/val`
     * `yolo_4cats/labels/train`, `yolo_4cats/labels/val`
   * Handle the case where some files are stored with uppercase `.JPG` by normalizing extensions.

4. **Generate `taco.yaml` for four-class dataset**

   * Write `yolo_4cats/taco.yaml` referencing the new paths and containing the list of four class names in the correct order.

5. **Visual checks**

   * As in the previous notebook, there are utilities to display a few random training images with their new bounding boxes and four-class labels overlaid, to sanity-check that remapping and conversions are correct.

**Outputs**

* `yolo_4cats/` with:

  * `images/train`, `images/val`
  * `labels/train`, `labels/val`
  * `taco.yaml` describing the four classes.

---

## 2_dataset_analysis 4cats.ipynb

**Goal:** Repeat the same style of analysis as `1_dataset_analysis.ipynb`, but on the new four-class dataset `yolo_4cats`, to verify that the regrouping and new split behave as expected.

It mirrors the structure of `1_dataset_analysis.ipynb`, with paths changed:

* Loads `yolo_4cats/taco.yaml`.
* Collects images and labels from `yolo_4cats/images/*` and `yolo_4cats/labels/*`.

Analyses performed:

1. **Class distribution (four classes only)**

   * Counts instances per class.
   * Checks that the four meta-classes are better balanced than the original fine-grained categories.

2. **Object size distribution**

   * Same `SMALL_THR` and `MEDIUM_THR` thresholds.
   * Summarizes proportion of small/medium/large objects across train and val.

3. **Image resolution and aspect ratio**

   * Basic stats and resolution frequency for the new dataset.

4. **Training image density**

   * Counts instances per training image.
   * Uses a `DENSE_THRESH` (e.g. 30 objects) to highlight very crowded images.
   * Produces a cumulative distribution plot of object counts and lists the 20 densest images.

This notebook ensures that the final four-class dataset is suitable for training and reveals any remaining imbalance or extreme edge cases.

---

## 3_custom_4cats_seg.ipynb

**Goal:** Create a segmentation version of the four-class dataset (`dataset_yolo_4cats_seg`) using polygon masks from the COCO annotations, and keep the same four high-level categories.

Configuration:

```python
COCO_JSON   = "TACO/data/annotations.json"
IMAGES_ROOT = "TACO/data"

YOLO_ROOT   = "dataset_yolo_4cats_seg"

TRAIN_PCT = 0.85
VAL_PCT   = 0.15
SEED      = 42
```

Conceptually similar to `2_custom_4cats.ipynb`, but with segmentation:

1. **Four meta-classes**

   * Same mapping to the four main categories:

     * Plastic
     * Paper
     * Glass
     * Unsorted

2. **Use segmentation polygons instead of boxes**

   * For each annotation, read the polygon segmentation from COCO.
   * Convert polygon coordinates into YOLO segmentation format:

     * class id
     * normalized bounding information
     * normalized polygon points.
   * Save one `.txt` label file per image in `labels/` using the YOLOv8 segmentation convention.

3. **Train/val split and file organization**

   * Split into train and val with the same percentages and seed.
   * Copy images to `dataset_yolo_4cats_seg/images/train` and `/val`.
   * Save corresponding segmentation labels in `labels/train` and `labels/val`.
   * Fix uppercase `.JPG` filenames where necessary.

4. **Generate `taco.yaml` for segmentation**

   * Write `dataset_yolo_4cats_seg/taco.yaml` with:

     * `path`: absolute dataset root
     * `train`, `val` paths
     * `names`: four class names.

5. **Dataset sanity checks**

   * Functions to visualize a few images overlayed with both bounding boxes and their polygon masks, confirming that:

     * the four-category mapping is correct,
     * polygon coordinates are aligned with the original images.

6. **Basic image counting**

   * A helper `count_images(folder)` to verify the number of training and validation images matches expectations.

**Outputs**

* `dataset_yolo_4cats_seg/` with:

  * `images/train`, `images/val`
  * `labels/train`, `labels/val` (segmentation labels)
  * `taco.yaml`

---