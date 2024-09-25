<h1 align="center">Classification of Dog and Cat Images via Convolutional Neural Networks</h1>

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
- **Color Mode:** RGB (3 channels)

## Data Preprocessing

To prepare the dataset for model training, several preprocessing steps are applied:

- **Resizing:** All images are resized to 128x128 pixels to ensure uniformity and reduce computational requirements.
- **Normalization:** Pixel values are scaled to a range of [0, 1] by dividing by 255, which facilitates better convergence during training.
- **Data Augmentation:** For the training set, data augmentation techniques are employed to enhance model robustness:
  - **Random Horizontal Flipping**
  - **Random Rotation:** Up to ±15 degrees
  - **Random Zooming:** Up to 10%

## Data Splitting

To effectively train and evaluate the model, the dataset is divided into three subsets:

**Training Dataset:**

- **Number of Images:** 20,000
  - **Dogs:** 10,000 images
  - **Cats:** 10,000 images
- **Path:** "./train/train"
- **Purpose:** Used to train the model by allowing it to learn features and patterns associated with each class.
- **Content:** Includes both the input images and their corresponding labels (dog or cat).

**Validation Dataset:**

- **Number of Images:** 5,000
  - **Dogs:** 2,500 images
  - **Cats:** 2,500 images
- **Path:** "./train/train"
- **Purpose:** Used to tune hyperparameters and make decisions about model architecture to prevent overfitting.
- **Content:** Includes input images and labels.

**Test Dataset:**

- **Number of Images:** 12,500
- **Path:** "./test1/test1"
- **Purpose:** Used to evaluate the final model performance. Participants will predict labels for these images.
- **Content:** Only input images are provided without labels to ensure an unbiased evaluation.

## Access to the Datasets

The dataset is hosted on a publicly accessible Google Drive folder to facilitate easy downloading:
- **Google Drive Link:** [Dog vs. Cat Dataset](https://drive.google.com/file/d/1PnSKt8yS87-a-v6NitmCfH_xg1p2YQju/view?usp=drive_link) </br>
*Note: Ensure you have sufficient storage space and a stable internet connection before downloading the dataset.*

## Installation

Implementation Details:
- Programming Language:
  - Python 3.9.19
- Libraries and Frameworks:
  - Tensorflow
  - Keras
  - NumPy
  - Pandas
  - Matplotlib
  - scikit-learn
- Environment:
  - Jupyter Notebook
- Hardware Requirements:
  - GPU acceleration recommended for faster training (e.g., NVIDIA GPU with CUDA support)

Setup environment
```shell script
# Clone this project repository under your workspace folder
git clone https://github.com/YoungJeansKR/XAI606
cd XAI606
# Create conda environment
conda create -n xai606 python=3.9.19
# Activate the working environment
conda activate xai606
# Install the packages
conda install tensorflow keras pandas numpy scikit-learn matplotlib pillow
```

## Training

```shell script
python train.py
```

## Evaluation

```shell script
python eval.py
```

## Contact

If you have any questions, please contact me at the email below. </br>
<yeongjinkim@korea.ac.kr>
