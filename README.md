# Question 1: Image Classification with SIFT & MLP

# Project Overview
This project involves image classification using the Bag of Visual Words (BoVW) method with SIFT features and a Multi-Layer Perceptron (MLP) for classification. The dataset consists of multiple image categories.

#Requirements
# Ensure you have the required dependencies installed before running the project:

- pip install numpy Pillow scikit-learn matplotlib cyvlfeat joblib

- Usage

- Prepare Dataset: Ensure the dataset is stored in the data/ directory with category-wise subfolders.

- Run the Pipeline:

- ASSIGNMENT_MAIN_2.ipynb
- Expected Output:

- Vocabulary generation using MiniBatchKMeans

- Feature extraction using SIFT

- MLP training with performance evaluation

- Confusion matrix and accuracy report

- Model saved as mlp_model.pkl, vocab.pkl, and preprocessor.pkl



## Question 2: Image Classification with PyTorch MLP

# Project Overview

- This project uses a three-layer MLP classifier in PyTorch to classify images. Images are loaded, preprocessed, and passed     through an optimized neural network.

- Requirements

- Ensure you have the required dependencies installed before running the project:

- pip install numpy torch torchvision matplotlib scikit-learn Pillow

# Usage

-Prepare Dataset: Ensure the dataset is stored in the data/ directory with category-wise subfolders.

- Run the Training Pipeline:

- ASSIGNMENT_MAIN_2.ipynb

- Expected Output:

- Image loading and preprocessing

- Training of MLP using PyTorch

- Accuracy and loss curves visualization

- Model evaluation on test data

- Visualization

- The script generates loss and accuracy plots to track training progress.

## Project Structure
.
├── data/
│   ├── agricultural/
│   ├── airplane/
│   ├── ...
│
├── question1.py  # SIFT + BoVW + MLP pipeline
├── question2.py  # PyTorch MLP pipeline
├── vocab.pkl  # Saved vocabulary
├── preprocessor.pkl  # Saved PowerTransformer
├── mlp_model.pkl  # Trained model (Q1)
└── README.md






   


