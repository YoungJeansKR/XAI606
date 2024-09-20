<h1 align="center">Classification of Dog and Cat Images via Convolutional Neural Networks</h1>

## Project Introduction

Accurate image classification is a fundamental challenge in computer vision, with profound implications for automation and artificial intelligence applications. This project focuses on developing a deep learning model, specifically a Convolutional Neural Network (CNN), to distinguish between images of dogs and cats with high precision. By leveraging a substantial and diverse dataset, the study aims to contribute to the advancement of image recognition technologies and provide a robust model that can be utilized in various real-world scenarios.

## Objective

The primary objective of this project is to develop a robust image classification model that can accurately distinguish between images of dogs and cats. By leveraging advanced machine learning techniques, specifically Convolutional Neural Networks (CNNs), we aim to achieve high accuracy in classifying these images, which can serve as a foundation for more complex image recognition tasks in the future.

## Motivation

The motivation behind this project stems from several key considerations:

- **Technological Advancement:** Enhancing image classification algorithms contributes to the broader field of computer vision, facilitating advancements in autonomous systems, robotics, and artificial intelligence.
- **Practical Applications:** Accurate classification models can be applied in numerous industries, including pet-related businesses, veterinary diagnostics, content filtering on social media platforms, and e-commerce, where automated image tagging and sorting are valuable.
- **Educational Contribution:** The project serves as an educational resource, offering insights into deep learning methodologies, data preprocessing techniques, and model evaluation strategies. It provides a practical framework for students and researchers to engage with contemporary machine learning challenges.

## Dataset Description

The dataset comprises a substantial collection of images of dogs and cats, sourced to represent a wide variety of breeds, poses, and environments. This diversity ensures that the model learns to generalize well to new, unseen data.

## Dataset Details

- **Total Number of Images:** 37,500
  - **Dog Images:** 18,750
  - **Cat Images:** 18,750
- **Image Format:** JPEG (.jpg)
- **Image Size:** Varied original sizes, standardized to 128x128 pixels during preprocessing
- **Color Mode:** RGB (3 channels)

## Data Splitting

To effectively train and evaluate the model, the dataset is divided into three subsets:

**Training Dataset:**

- **Number of Images:** 20,000
  - **Dogs:** 10,000 images
  - **Cats:** 10,000 images
- **Purpose:** Used to train the model by allowing it to learn features and patterns associated with each class.
- **Content:** Includes both the input images and their corresponding labels (dog or cat).

**Validation Dataset:**

- **Number of Images:** 5,000
  - **Dogs:** 2,500 images
  - **Cats:** 2,500 images
- **Purpose:** Used to tune hyperparameters and make decisions about model architecture to prevent overfitting.
- **Content:** Includes input images and labels.

**Test Dataset:**

- **Number of Images:** 12,500
- **Purpose:** Used to evaluate the final model performance. Participants will predict labels for these images.
- **Content:** Only input images are provided without labels to ensure an unbiased evaluation.

## Access to the Datasets

The dataset is hosted on a publicly accessible Google Drive folder to facilitate easy downloading:
- **Google Drive Link:** [Dog vs. Cat Dataset](https://drive.google.com/file/d/1PnSKt8yS87-a-v6NitmCfH_xg1p2YQju/view?usp=drive_link) </br>
*Note: Ensure you have sufficient storage space and a stable internet connection before downloading the dataset.*

## Instructions for Participants

- **Training and Validation:**
  - Utilize the provided training and validation datasets to develop and fine-tune your image classification models.
  - Experiment with different architectures, hyperparameters, and preprocessing techniques to optimize performance.
- **Testing:**
  - Apply your trained model to the test dataset to generate predictions.
  - Submit your predicted labels for the test images in the specified format (e.g., a CSV file with image filenames and predicted labels).
- **Evaluation Metrics:**
  - **Accuracy:** The primary metric for evaluating model performance.
  - **Confusion Matrix:** To understand the types of errors made by the model.
  - **Precision and Recall:** For a more detailed performance analysis, especially if classes are imbalanced.

## Baseline Model Implemetation

Implementation Details:
- Programming Language:
  - Python 3.8
- Libraries and Frameworks:
  - Pytorch 2.12
  - Keras
  - NumPy
  - Pandas
  - Matplotlib
  - scikit-learn
- Environment:
  - Jupyter Notebook or any Python IDE
- Hardware Requirements:
  - GPU acceleration recommended for faster training (e.g., NVIDIA GPU with CUDA support)

Requirements:
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.8.2.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenAI API (optional, if you want to construct hierarchies using LMMs)

Setup environment
```shell script
# Clone this project repository under your workspace folder
git clone https://github.com/YoungJeansKR/XAI606
cd XAI606
# Create conda environment and install the dependencies
conda env create -n XAI606 -f XAI606.yml
# Activate the working environment
conda activate XAI606
```
