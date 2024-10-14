# Grape Disease Detection

This project focuses on developing a deep learning-based image classification model to detect grape diseases from leaf images. The goal is to support early disease detection in vineyards, allowing for better crop management and improved yields.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Streamlit Web Application](#streamlit-web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
Grape disease detection is critical to prevent damage in vineyards, improve crop yields, and ensure the quality of produce. This project uses a Convolutional Neural Network (CNN) to classify grape leaf images into different categories of diseases, such as:
- Black rot
- Esca
- Leaf Blight
- Healthy leaves

This automated approach to disease detection reduces manual inspections and enables early intervention for more efficient vineyard management.

## Dataset
The dataset consists of grape leaf images categorized into multiple disease classes, including healthy samples. Some preprocessing steps include:
- **Image Augmentation**: To address class imbalance and improve generalization.
- **Image Resizing**: All images are resized to 170x170 pixels.
- **Normalization**: Pixel values are normalized between 0 and 1 to speed up training.
- **Data Split**: 80% of the data was used for training, and 20% was reserved for testing.

## Model Architecture
The model is a Convolutional Neural Network (CNN), well-suited for image classification tasks. The architecture includes:
- **Convolutional Layers**: To extract spatial features from images.
- **Activation Function**: ReLU introduces non-linearity to the model.
- **Max-Pooling Layers**: For down-sampling.
- **Flattening Layer**: To convert 2D feature maps to a 1D vector.
- **Dense Layers**: For classification.
- **Optimizer**: Adam.
- **Loss Function**: Categorical Cross-Entropy.

### Hyperparameters:
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy
- Early stopping was applied to avoid overfitting.

## Results
The model achieved the following performance:
- **Training Accuracy**: 95%
- **Validation Accuracy**: 97%

## Streamlit Web Application
A web application was developed using Streamlit, allowing users to upload grape leaf images and receive predictions on the type of disease. The app predicts whether the leaf is healthy or affected by diseases such as Black Rot, Esca, or Leaf Blight.

The app is hosted on Hugging Face Spaces. You can access the application and try it out here:
👉 [Grape Disease Detection Web App](https://huggingface.co/spaces/poluhamdi/GrapeDiseaseDetector)

You can also explore the project's code on Kaggle:
👉 [Kaggle Notebook](https://www.kaggle.com/code/hamdipolu/grape-disease-detection)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/grape-disease-detection.git
    cd grape-disease-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the `data/` folder.

## Usage
To run the Streamlit web application locally, use the following command:
```bash
streamlit run app.py
```

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
