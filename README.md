# Rare Species Image Classification

A deep learning computer vision project for classifying rare species images into **202 taxonomic families** across **5 phyla** using transfer learning, image preprocessing, outlier detection, and class imbalance mitigation.

This project was completed as part of a Deep Learning course project and focuses not only on model performance, but also on the full machine learning workflow: data exploration, cleaning, preprocessing, model selection, tuning, evaluation, and error analysis.

## Project Overview

The goal of this project was to build an image classification model capable of identifying rare species families from a dataset of approximately **12,000 images** sourced from the Encyclopedia of Life.

The task was challenging because the dataset contained:

- Severe class imbalance across phyla and families
- Noisy images such as sketches, X-rays, diagrams, signs, maps, and text documents
- High visual similarity between closely related taxonomic families
- Multi-subject images and ambiguous cases
- A large number of classes relative to the dataset size

The final model used **EfficientNetB4** as a frozen feature extractor with a tuned classification head and achieved strong performance on the test set.

## Key Results

| Metric | Score |
|---|---:|
| Test Accuracy | 84.01% |
| Micro F1-Score | 84.01% |
| Macro F1-Score | 82.88% |
| Weighted F1-Score | 83.99% |
| Test Loss | 0.5643 |

A metadata-enhanced inference experiment using known phylum information improved accuracy slightly from **84.01% to 84.35%**.

## Main Techniques Used

### Data Exploration

The original dataset contained **11,983 images** distributed across 5 phyla and 202 families. The data was highly imbalanced, with Chordata representing the majority of the dataset and Echinodermata being heavily underrepresented.

Initial image analysis was used to select a working resolution of **500x375 pixels**, balancing image detail with computational efficiency.

### Data Cleaning and Outlier Detection

A major part of the project involved identifying and removing unsuitable or noisy images.

Several approaches were tested:

- **YOLOv8 person detection**  
  Tested to detect images containing humans, but rejected because many valid images contained researchers holding animals.

- **ImageNet-based classification**  
  Tested using InceptionV3, but rejected because it incorrectly flagged too many valid wildlife images as non-animal images.

- **CLIP semantic scoring**  
  Used to compare each image against “good” wildlife prompts and “bad” prompts such as X-rays, diagrams, text documents, food, and humans. This successfully identified many high-confidence outliers.

- **Distance-to-centroid filtering**  
  EfficientNetB4 embeddings were used to calculate each image’s distance from its class centroid. Images far from their class centroid were removed as likely outliers.

After preprocessing, the dataset was reduced from **11,983 images** to **10,140 images**.

Final split:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 7,780 | 77% |
| Validation | 865 | 9% |
| Test | 1,495 | 15% |

### Data Augmentation

Aggressive brightness, contrast, and colour augmentations were tested but rejected after visual inspection because they created unrealistic images.

The final augmentation strategy used conservative geometric transformations:

- Horizontal flip
- Rotation up to ±15 degrees
- Zoom up to ±10%

This helped introduce useful variation without distorting the biological features of the species.

## Model Development

Multiple model approaches were tested before selecting the final architecture.

| Model | Validation Accuracy | Notes |
|---|---:|---|
| Custom CNN from scratch | ~2.5% | Near random chance |
| EfficientNetB0 + Dense layers | ~63% | Baseline transfer learning |
| EfficientNetB0 + tuned head | ~79% | Improved after hyperparameter tuning |
| EfficientNetB4 + classification layer only | ~83% | Strong but rejected due to limited trainable non-linearity |
| EfficientNetB4 + tuned dense head | ~84% | Final selected model |

The custom CNN performed poorly because the dataset was too small and complex to train a deep image classifier from scratch. Transfer learning was therefore much more effective.

## Final Architecture

The final model architecture was:

```text
EfficientNetB4 (frozen)
→ GlobalAveragePooling2D
→ BatchNormalization
→ Dense(600)
→ Dropout(0.4)
→ Dense(202, softmax)
