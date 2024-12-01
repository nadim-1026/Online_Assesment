# Disease Classification Using Chest X-Ray Images: A Comprehensive Deep Learning Approach

This repository contains the implementation of a deep learning-based approach for multi-class classification of chest X-ray images into four categories: **COVID-19**, **Normal**, **Lung Opacity**, and **Viral Pneumonia**. The project explores various pre-trained and custom models, with a fine-tuned VGG16 model achieving the highest accuracy of **94%** on an augmented dataset.

## Overview
Chest X-rays are crucial for diagnosing lung-related diseases, and this project aims to automate disease detection using deep learning. The pipeline involves preprocessing, model selection, training, evaluation, and hyperparameter tuning to create a reliable classification model.

## Dataset
The dataset used is the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), containing **21,165 chest X-ray images** across four classes:
- **COVID-19**: 3,616 images
- **Normal**: 10,192 images
- **Lung Opacity**: 6,012 images
- **Viral Pneumonia**: 1,345 images

### Preprocessing
1. **Resizing**: Images resized to 224x224 pixels.
2. **Normalization**: Based on ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
3. **Balancing and Augmentation**:
   - Phase 1: Downsampled dataset (1,345 images per class).
   - Phase 2: Augmented dataset (4,035 images per class) using techniques like random flips, rotation, and color jitter.
4. **Train-Validation-Test Split**: 60%-20%-20%.

## Models Evaluated
1. **ResNet18**: Struggled with underfitting.
2. **Customized ResNet18**: Minor improvements with added layers.
3. **Custom CNN**: Faced overfitting issues.
4. **Fine-Tuned VGG16**: Achieved **94% accuracy** with the following enhancements:
   - Pre-trained convolutional layers from ImageNet.
   - Custom fully connected layers with dropout for regularization.
   - StepLR learning rate scheduler for smooth convergence.

## Training Process
### Two-Stage Training Strategy
1. **Phase 1**: Training on a balanced dataset to address class imbalance.
2. **Phase 2**: Fine-tuning on an augmented dataset to enhance generalization.

### Hyperparameters
- Optimizer: Adam (`lr=0.001`)
- Loss Function: Cross-Entropy Loss
- Learning Rate Scheduler: StepLR (halves learning rate every 5 epochs)
- Dropout: 0.5
- Batch Size: 32

## Results
| Metric       | ResNet18 | Custom ResNet18 | Custom CNN | Fine-Tuned VGG16 |
|--------------|----------|-----------------|------------|------------------|
| Accuracy     | 87.55%   | 91.73%          | 89.30%     | **94%**          |
| Precision    | 89.30%   | 91.82%          | 90.50%     | **94.60%**       |
| Recall       | 87.88%   | 91.87%          | 89.75%     | **94.50%**       |
| F1-Score     | 87.55%   | 91.79%          | 89.90%     | **94.53%**       |

The confusion matrix for the fine-tuned VGG16 model shows minimal misclassification across all four classes.
| Confusion Matrix Balanced Dataset  | Confusion Matrix Augumented Dataset |
|------------------------------------|-------------------------------------|                                                                           
| ![confusion matrix of Balanced Dataset](https://github.com/user-attachments/assets/9f394bd5-6261-49f1-8ff9-8e65706db865)    | ![confusionmatrix for Augumented Dataset](https://github.com/user-attachments/assets/1975376c-3f68-4e3a-a34c-006829fd0456)  |






## Key Challenges
1. **Class Imbalance**: Solved using downsampling and augmentation.
2. **Overfitting**: Addressed via dropout layers and increased dataset variability.
3. **Underfitting**: Resolved by fine-tuning a deeper model (VGG16).

## Conclusion
This project highlights the effectiveness of fine-tuned deep learning models for automated disease classification using chest X-rays. The VGG16 model achieved superior performance, making it a reliable tool for medical diagnostics.

## References
- [COVID-19 Radiography Database - Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- COVID-CXR GitHub Repository
- Radiopaedia - COVID-19 Chest Imaging
- Stack Overflow (for debugging)
- ChatGPT (conceptualization and debugging)

---

