# Skin Cancer Classification Using MLP and CNN

## Overview
This project focuses on the classification of skin cancer images using deep learning models, specifically Multilayer Perceptrons (MLP) and Convolutional Neural Networks (CNN) [cite: 118, 119, 591]. The goal is to accurately classify images of skin lesions into their corresponding classes to aid in early detection [cite: 590, 591, 592].

## Dataset
The project utilizes the **DermaMNIST dataset**, which contains preprocessed medical images of skin lesions stored in `.npz` format [cite: 119, 591, 593, 594]. 
* **Classes:** The dataset consists of 7 distinct classes of skin lesions [cite: 592].
* **Image Properties:** Each image is 28x28 pixels with 3 color channels (RGB) [cite: 626].
* **Imbalance:** The dataset is highly imbalanced; for example, Class 5 dominates with roughly 70% of the total dataset, while classes like 3 and 6 are extremely underrepresented [cite: 628, 629].

## Methodology

### 1. Data Preprocessing
* **Dimensionality Reduction:** Images were flattened into 1D vectors for MLP inputs [cite: 127, 486]. Principal Component Analysis (PCA) was also applied to reduce feature dimensionality (e.g., to 10, 15, 20, or 100 components) to minimize noise and computation time [cite: 503, 504].
* **Class Imbalance Handling:** * For the MLP models, `compute_class_weight` from `sklearn` was used to penalize underrepresented classes [cite: 64, 488, 489].
  * For the CNN model, a combination of **Over-Sampling** (duplicating minority classes) and **Under-Sampling** (reducing majority classes) was employed to balance the training data [cite: 607, 608].
* **Data Augmentation:** The `ImageDataGenerator` was used in the CNN pipeline to generate new image variations, helping to reduce overfitting and improve model generalization [cite: 595, 612].

### 2. Models Evaluated
* **Multilayer Perceptron (MLP):** Various architectures were tested, ranging from 1 to 7 dense layers, utilizing Dropout and Batch Normalization to reduce overfitting [cite: 103, 433, 477, 577].
* **MLP with PCA:** The MLP models were re-evaluated using PCA-reduced inputs to analyze the effects of dimensionality reduction on performance [cite: 502, 577].
* **Convolutional Neural Network (CNN):** A deep CNN architecture was constructed using 4 convolutional blocks (32 → 64 → 128 → 256) followed by 3 fully connected layers (256 → 128 → 64) [cite: 616]. A dynamic learning rate scheduler (`ReduceLROnPlateau`) was used to improve training stability [cite: 597, 614, 615].

## Key Results

* **Standard MLP:** The baseline MLP models generally plateaued at a testing accuracy of approximately **66.88%** [cite: 214, 299, 416].
* **MLP with PCA (Best Model):** The most effective and efficient model was **Trial 11**, a 3-Layer MLP using 100 PCA components. This model achieved the highest testing accuracy of **73.37%** with only ~217k parameters, offering a great balance of performance and efficiency [cite: 572, 577, 578].
* **CNN:** The CNN approach showed strong generalization, reaching a top testing accuracy of **71.52%** with a dynamically reduced learning rate [cite: 651, 652].

## Dependencies
The project code relies on the following major Python libraries:
* `numpy` [cite: 14]
* `pandas` [cite: 594]
* `tensorflow` / `keras` [cite: 16, 594]
* `scikit-learn` (`sklearn`) [cite: 17]
* `matplotlib` / `seaborn` [cite: 15, 598]
* `imbalanced-learn` (`imblearn`) [cite: 607]

## Disclaimer
This project is for educational and research purposes only and is not intended for clinical use.
