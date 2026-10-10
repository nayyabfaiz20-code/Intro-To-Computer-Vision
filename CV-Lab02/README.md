# Computer Vision — Image Filtering & Transfer Learning Experiment

A deep learning-based **Computer Vision image classification project** that investigates how different image preprocessing and filtering techniques affect the performance of pretrained CNN models.

The project compares **EfficientNetB0, ResNet50, and ResNet18** using a baseline (unfiltered) condition and multiple image-filtering techniques. Performance is evaluated using several classification metrics and visual analysis.

---

## Project Overview

The objective of this project is to study the impact of image filtering techniques on deep learning-based image classification.

The experiment follows a controlled setup where the same:

* Dataset
* Image size
* Data augmentation
* Batch size
* Class weights
* Training strategy
* Number of epochs
* Evaluation metrics

are maintained across experiments, while the **image filtering condition is changed**.

This allows the effect of preprocessing/filtering to be analyzed more fairly.

---

##  Objectives

The main objectives of this project are:

1. Train pretrained deep learning models for image classification.
2. Establish an unfiltered baseline.
3. Apply different image filtering techniques.
4. Compare model performance under each filtering condition.
5. Analyze the effect of filtering on individual classes.
6. Evaluate models using multiple performance metrics.
7. Generate confusion matrices and training curves.
8. Identify how preprocessing affects different CNN architectures.

---

##  Models Used

Three pretrained CNN architectures are evaluated:

### 1. EfficientNetB0

EfficientNetB0 is a lightweight convolutional neural network that provides a good balance between computational efficiency and classification performance.

### 2. ResNet50

ResNet50 is a 50-layer residual neural network that uses residual connections to improve the training of deeper networks.

### 3. ResNet18

ResNet18 is a relatively lightweight residual network implemented using PyTorch and torchvision.

---

##  Image Filtering Techniques

The notebook evaluates multiple filtering conditions in addition to the original unfiltered images.

The filtering pipeline includes:

* **Baseline / Original Image**
* **Gaussian Filtering**
* **Median Filtering**
* **Sharpening**
* **Sobel Edge Filtering**
* **Additional configured filtering condition**

The purpose is to determine whether removing noise, enhancing edges, or modifying image details improves or reduces classification performance.

---

##  Transfer Learning

The project uses pretrained ImageNet models as feature extractors.

For the TensorFlow/Keras models, the pretrained backbone is followed by a custom classification head:

```text
Input Image
     ↓
Pretrained CNN Backbone
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Dense Layer (256)
     ↓
Dropout
     ↓
Output Classification Layer
```

The final classification layer is adapted to the selected number of classes.

---

##  Data Processing Pipeline

The overall pipeline is:

```text
Dataset
   ↓
Image Loading
   ↓
Image Resizing (224 × 224)
   ↓
Image Filtering
   ↓
Normalization
   ↓
Data Augmentation
   ↓
Pretrained CNN
   ↓
Classification
   ↓
Performance Evaluation
```

Training data uses augmentation techniques such as:

* Rotation
* Width shifting
* Height shifting
* Zooming
* Horizontal flipping

Validation and test images are processed without training augmentation.

---

##  Evaluation Metrics

Each model/filter combination is evaluated using:

| Metric            | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| Accuracy          | Overall percentage of correctly classified images             |
| Balanced Accuracy | Average recall across classes, useful for imbalanced datasets |
| Macro Precision   | Average precision across all classes                          |
| Macro Recall      | Average recall across all classes                             |
| Macro-F1          | Average F1-score across all classes                           |
| Macro-AUC         | Multi-class ROC-AUC averaged across classes                   |

The project also generates **classification reports** containing per-class:

* Precision
* Recall
* F1-score

---

##  Confusion Matrix

Confusion matrices are generated for every model and filtering condition.

They help analyze:

* Correct predictions
* Misclassified samples
* Confusion between individual classes
* The effect of filtering on particular classes

Example output files include:

```text
cm_EfficientNetB0_*.png
cm_ResNet50_*.png
cm_ResNet18_*.png
```

---

##  Training Curves

Training and validation curves are generated for each experiment.

The project visualizes:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

Example:

```text
Training Accuracy
        ↗
       /
      /
-----/------------ Epochs

Validation Accuracy
        ↗
       /
------/----------- Epochs
```

Generated curve files follow the format:

```text
curves_EfficientNetB0_*.png
curves_ResNet50_*.png
curves_ResNet18_*.png
```

---

##  Experimental Design

The experiment consists of:

```text
3 Models × 6 Conditions = 18 Experiments
```

Each model is evaluated using:

```text
Baseline
   +
Filter 1
   +
Filter 2
   +
Filter 3
   +
Filter 4
   +
Filter 5
```

The main purpose is to compare the performance of each model under identical experimental settings.

---

##  Project Structure

A recommended GitHub repository structure is:

```text
Computer-Vision-Filtering-Experiment/
│
├── CV.ipynb
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   └── README.md
│
├── results/
│   ├── task2_comparison_table.csv
│   ├── task2_all_results.json
│   ├── comparative_accuracy_by_filter.png
│   ├── filter_examples.png
│   │
│   ├── confusion_matrices/
│   │   └── *.png
│   │
│   └── training_curves/
│       └── *.png
│
└── screenshots/
    └── project_output.png
```

> **Note:** The original dataset does not need to be uploaded to GitHub if it is large or subject to licensing restrictions. You can provide instructions for obtaining or placing the dataset in the `dataset/` directory.

---
##  Technologies & Libraries

### Programming Language

* Python 3.x

### Deep Learning

* TensorFlow
* Keras
* PyTorch
* Torchvision

### Computer Vision

* OpenCV

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib

### Development Environment

* Google Colab
* Jupyter Notebook

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Computer-Vision-Filtering-Experiment.git
```

Move into the project directory:

```bash
cd Computer-Vision-Filtering-Experiment
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Option 1 — Google Colab

The easiest way to run the project is using Google Colab.

1. Open `CV.ipynb`.
2. Upload/open the notebook in Google Colab.
3. Make sure the dataset path is correctly configured.
4. Select a GPU runtime.
5. Run the notebook cells sequentially.

For faster training:

```text
Runtime → Change runtime type → GPU
```

---

### Option 2 — Jupyter Notebook

Install Jupyter:

```bash
pip install notebook
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
CV.ipynb
```

Then execute the notebook cells in order.

---

## ⚙️ Experiment Configuration

The main experiment uses:

```python
EPOCHS = 5
```

Images are resized to:

```text
224 × 224
```

The notebook uses pretrained ImageNet weights and applies class weighting to address class imbalance.

> Training all model/filter combinations can require significant computational time. A GPU-enabled environment such as Google Colab is recommended.

---

##  Generated Results

After completing the notebook, several result files are generated.

### Comparison Table

```text
task2_comparison_table.csv
```

Contains the performance of all model/filter combinations.

### Raw Results

```text
task2_all_results.json
```

Stores the experiment results in JSON format.

### Filtering Examples

```text
filter_examples.png
```

Shows an example image under the different filtering conditions.

### Confusion Matrices

Generated for each model and filtering condition.

### Training Curves

Generated for each model and filtering condition.

### Comparative Accuracy

```text
comparative_accuracy_by_filter.png
```

Provides a visual comparison of accuracy across filtering conditions and models.

---

##  Analysis

The notebook investigates the following questions:

### 1. Baseline Performance

How well does each pretrained model perform without image filtering?

### 2. Effect of Filtering

Does applying an image filter improve or decrease classification performance?

### 3. Model-Specific Response

Do different CNN architectures respond differently to the same filtering technique?

### 4. Class-Level Performance

Which classes experience the largest changes in:

* Precision
* Recall
* F1-score

### 5. Balanced Performance

Does filtering improve or reduce:

* Balanced Accuracy
* Macro-F1
* Macro-AUC

These analyses help determine whether image preprocessing is beneficial for the classification task.

---

##  Key Findings

The final conclusions should be based on the results generated by `CV.ipynb`.

The notebook provides the required comparison table and visualizations so that the performance of every model/filter combination can be examined.

Instead of assuming that a particular filter is universally beneficial, the analysis considers the response of each individual architecture and class.

---

##  Reproducibility

To reproduce the experiments:

1. Use the same dataset.
2. Use the same selected classes.
3. Use the same image size.
4. Use the same train/validation/test split.
5. Use the same augmentation settings.
6. Use the same class weights.
7. Use the same number of epochs.
8. Run all model/filter combinations.
9. Generate the comparison table and visualizations.

---

##  Future Improvements

Possible future improvements include:

* Fine-tuning pretrained CNN layers
* Testing additional CNN architectures
* Testing additional image enhancement techniques
* Hyperparameter optimization
* Cross-validation
* Ensemble learning
* Explainable AI using Grad-CAM
* Automated experiment tracking
* GPU/TPU optimization
* Web-based image classification interface
* Deployment as an interactive AI application

---

##  License

This project is intended for **educational and research purposes**.

If the dataset used in this project has separate licensing or usage requirements, please follow the original dataset's license and terms.

---

##  Acknowledgments

This project uses pretrained deep learning architectures and open-source Python libraries for computer vision, deep learning, data processing, and evaluation.

Special thanks to the developers and research communities behind:

* TensorFlow / Keras
* PyTorch
* Torchvision
* OpenCV
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
