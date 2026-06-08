# Rare Species Image Classification

A deep learning computer vision project for classifying rare species images into **202 taxonomic families** across **5 phyla** using transfer learning, outlier detection, class imbalance handling, and model evaluation.

This project was completed as part of a Deep Learning course project. The aim was to build a robust image classification pipeline for a noisy, imbalanced, real-world biological image dataset sourced from the Encyclopedia of Life.

---

## Project Overview

The task was to classify rare species images into their correct taxonomic family.

The dataset contained approximately **12,000 images**, covering:

- **202 taxonomic families**
- **5 biological phyla**
- Highly imbalanced class distributions
- Noisy and inconsistent image types
- Visually similar species families
- Images containing sketches, X-rays, maps, text documents, signs, humans, and other non-animal content

The project focused not only on achieving high model accuracy, but also on building a complete deep learning workflow, including:

- Data exploration
- Image preprocessing
- Outlier detection
- Data augmentation
- Transfer learning
- Hyperparameter tuning
- Class imbalance mitigation
- Model evaluation
- Error analysis

---

## Key Results

The final model achieved strong performance on the held-out test set.

| Metric | Score |
|---|---:|
| Test Accuracy | **84.01%** |
| Micro F1-Score | **84.01%** |
| Macro F1-Score | **82.88%** |
| Weighted F1-Score | **83.99%** |
| Test Loss | **0.5643** |

A bonus metadata-enhanced inference experiment using known phylum information improved performance slightly:

| Version | Accuracy | Macro F1 |
|---|---:|---:|
| Base model | 84.01% | 82.88% |
| Phylum-constrained inference | 84.35% | 82.95% |

---

## Dataset Challenges

This was a difficult multi-class classification problem because of several real-world dataset issues.

### 1. Severe Class Imbalance

The dataset was highly imbalanced across both phyla and families.

- Chordata represented the large majority of the dataset.
- Echinodermata was heavily underrepresented.
- At family level, the largest family had roughly 10 times more images than the smallest families.

### 2. Noisy Image Data

The dataset included many images that were not useful for species classification, such as:

- X-rays
- Diagrams
- Sketches
- Maps
- Food images
- Signs
- Text documents
- Images containing humans without clearly visible animals

### 3. High Visual Similarity

Many classification errors came from families that were biologically and visually similar. Some cases were also ambiguous even from a human perspective.

---

## Project Workflow

```text
Raw Images
   ↓
Exploratory Data Analysis
   ↓
Image Format Cleaning
   ↓
CLIP Semantic Outlier Detection
   ↓
Train/Test Split
   ↓
EfficientNetB4 Embedding Extraction
   ↓
Distance-to-Centroid Outlier Detection
   ↓
Train/Validation Split
   ↓
Data Augmentation
   ↓
Transfer Learning Model Training
   ↓
Fine-Tuning
   ↓
Evaluation and Error Analysis
```

---

## Data Preprocessing

The original dataset contained **11,983 images**.

Several preprocessing steps were applied before model training.

### Image Format Cleaning

Unsuitable image formats were removed or converted:

- Removed grayscale images, mostly sketches and diagrams.
- Removed most CMYK images, which were mainly skull X-rays.
- Retained and converted only two valid CMYK photographs to RGB.

### Final Image Size

Images were resized to:

```text
500 x 375 pixels
```

This resolution was chosen because most images followed a similar 4:3 aspect ratio, while also balancing visual detail and computational efficiency.

---

## Outlier Detection

A major part of the project involved detecting and removing unsuitable images.

Several approaches were tested.

### YOLOv8 Person Detection

YOLOv8 was tested to identify images containing humans. However, this approach was rejected because it often flagged valid wildlife images where researchers were holding animals.

### ImageNet Classification

InceptionV3 pretrained on ImageNet was tested to identify non-animal images. This approach was also rejected because it flagged too many valid rare species images as outliers due to domain mismatch.

### CLIP Semantic Scoring

CLIP was used to compare each image against two groups of text prompts:

**Good prompts:**

- Animal photograph
- Wildlife photograph
- Species image
- Natural animal image

**Bad prompts:**

- X-ray
- Diagram
- Text document
- Food
- Human
- Map
- Sign

Images where the bad-prompt similarity was higher than the good-prompt similarity were flagged as outliers.

This method successfully removed many clearly unsuitable images, including X-rays, diagrams, text-heavy images, and irrelevant content.

### Distance-to-Centroid Outlier Detection

EfficientNetB4 embeddings were generated for the images. For each class, a centroid was calculated using the training images. Images far away from their class centroid were treated as potential outliers.

Two filtering passes were applied:

1. Remove images beyond the 95th percentile distance.
2. Remove remaining extreme images beyond the 97.5th percentile distance.

This helped remove additional unusual or mislabeled examples that CLIP did not catch.

---

## Final Dataset Split

After preprocessing and outlier removal, **10,140 images** remained.

| Split | Images | Percentage |
|---|---:|---:|
| Training | 7,780 | 77% |
| Validation | 865 | 9% |
| Test | 1,495 | 15% |

The splits were stratified to preserve class distributions as much as possible.

---

## Data Augmentation

Aggressive augmentation was tested, including:

- Brightness shifts
- Contrast changes
- Colour channel adjustments

However, these produced unrealistic images and risked adding noise to the training process.

The final augmentation strategy used conservative geometric transformations:

- Horizontal flip
- Rotation up to ±15 degrees
- Zoom up to ±10%

This was chosen because animal orientation, camera angle, and distance can naturally vary without changing the species identity.

---

## Model Development

Several model approaches were tested before selecting the final architecture.

| Model | Validation Accuracy | Notes |
|---|---:|---|
| Custom CNN from scratch | ~2.5% | Near random chance |
| EfficientNetB0 + dense layers | ~63% | Initial transfer learning baseline |
| EfficientNetB0 + tuned head | ~79% | Improved after hyperparameter tuning |
| EfficientNetB4 + classification layer only | ~83% | Strong, but limited trainable non-linearity |
| EfficientNetB4 + tuned dense head | ~84% | Final selected model |

The custom CNN performed poorly because the dataset was too small and complex to train a deep model from scratch across 202 classes.

Transfer learning was therefore much more effective.

---

## Final Model Architecture

The final model used **EfficientNetB4** as a frozen feature extractor, followed by a custom classification head.

```text
EfficientNetB4
→ GlobalAveragePooling2D
→ BatchNormalization
→ Dense(600)
→ Dropout(0.4)
→ Dense(202, softmax)
```

Model size:

| Parameter Type | Count |
|---|---:|
| Total parameters | 18,878,193 |
| Trainable parameters | 1,200,786 |

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **Keras Tuner with Hyperband**.

The search space included:

- Number of dense layers
- Number of hidden units
- Dropout rate
- Optimiser choice

The best configuration was:

| Hyperparameter | Selected Value |
|---|---:|
| Dense layers | 1 |
| Hidden units | 600 |
| Dropout | 0.4 |
| Optimiser | Adagrad |
| Learning rate | 0.01 |

Adagrad was selected because it performed well for sparse and imbalanced class learning.

---

## Class Imbalance Handling

The dataset had large differences in class sizes, with some families containing hundreds of images and others containing very few.

To reduce the impact of imbalance, balanced class weights were computed using scikit-learn and passed into the model during training.

This increased the penalty for misclassifying minority classes and helped improve macro-level performance.

---

## Training Procedure

The model was trained in two stages.

### Stage 1: Main Training

- EfficientNetB4 frozen
- Custom classification head trainable
- Optimiser: Adagrad
- Learning rate: 0.01
- Early stopping used to prevent overfitting
- ReduceLROnPlateau used to lower the learning rate when validation performance stopped improving

### Stage 2: Fine-Tuning

- Last 5 layers of EfficientNetB4 unfrozen
- Batch normalization layers kept frozen
- Very low learning rate: 1e-7
- Fine-tuning improved validation accuracy slightly

---

## Error Analysis

Most misclassifications occurred in difficult cases, such as:

- Visually similar families within the same phylum
- Images containing multiple animals
- Unusual poses
- Occlusion
- Ambiguous or low-quality images
- Species with very similar morphology

The errors suggested that the model was not failing systematically. Instead, many mistakes came from genuinely difficult examples in the dataset.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB4
- Keras Tuner
- scikit-learn
- OpenAI CLIP
- Hugging Face Transformers
- YOLOv8 / Ultralytics
- NumPy
- Pandas
- PIL
- Matplotlib
- Seaborn
- Plotly

---

## Repository Structure

```text
rare-species-image-classification/
│
├── notebooks/
│   └── rare_species_classification.ipynb
│
├── reports/
│   └── project_report.pdf
│
├── figures/
│   ├── outlier_examples.png
│   ├── augmentation_examples.png
│   ├── confusion_matrix.png
│   └── misclassified_examples.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rare-species-image-classification.git
cd rare-species-image-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/rare_species_classification.ipynb
```

---

## Dataset Access

The raw image dataset is not included in this repository due to file size and licensing considerations.

The original images were sourced from the Encyclopedia of Life. This repository contains the project code, methodology, report, and results summary.

---

## Skills Demonstrated

This project demonstrates experience with:

- Deep learning for image classification
- Transfer learning
- Computer vision preprocessing
- Real-world noisy image datasets
- Outlier detection
- CLIP-based semantic filtering
- Embedding-based anomaly detection
- Class imbalance handling
- Hyperparameter tuning
- Model evaluation using F1-scores
- Error analysis
- Scientific reporting
- End-to-end machine learning workflow design

---

## Future Improvements

Potential improvements include:

- Using BioCLIP or other biodiversity-specific foundation models
- Applying autoencoders or isolation forests for more advanced anomaly detection
- Using object detection to isolate individual animals in multi-subject images
- Collecting additional images for underrepresented families
- Applying synthetic data generation for rare classes
- Testing Vision Transformers or ConvNeXt architectures
- Fine-tuning more EfficientNet layers with stronger regularisation

---

## Authors

Group 12:

- Yan Sidoryk
- Henry Lewis
- Abdul Rehman Khan
- Lowie De Wever

---

## Project Status

Completed as part of a Deep Learning course project.

The final model achieved approximately **84% test accuracy** and **0.84 weighted F1-score** on a challenging 202-class rare species classification task.

---

## License

This repository is intended for educational and portfolio purposes.
