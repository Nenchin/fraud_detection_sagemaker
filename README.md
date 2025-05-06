# Fraud Detection with Amazon SageMaker

This project implements a fraud detection model using Amazon SageMaker. The model is designed to identify fraudulent transactions based on a dataset of financial transactions.

## Project Structure

```
fraud-detection-sagemaker
├── notebooks
│   └── sagemaker_arch.ipynb       # Jupyter notebook containing the architecture for the fraud detection model
├── scripts
│   └── fraud_detection.py          # Python script for model training and evaluation
├── data
│   ├── train_set.csv               # Training dataset for the model
│   └── test_set.csv                # Test dataset for evaluating the model
├── requirements.txt                # List of Python dependencies
└── README.md                       # Project documentation
```

## Overview

The `sagemaker_arch.ipynb` notebook includes the following key components:
- Data preprocessing steps to clean and prepare the dataset.
- Exploratory data analysis (EDA) to understand the data distribution and relationships.
- Model training using a RandomForestClassifier.
- Deployment of the trained model to an Amazon SageMaker endpoint.

The `fraud_detection.py` script contains functions for:
- Loading the training and test datasets.
- Training the RandomForestClassifier model.
- Saving the trained model for future use.

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/Nenchin/fraud_detection_sagemaker.git
   cd fraud-detection-sagemaker
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

- Open the Jupyter notebook `notebooks/sagemaker_arch.ipynb` to run the entire workflow for fraud detection.
- Use the `scripts/fraud_detection.py` script to train the model and evaluate its performance on the test dataset.

## License

This project is licensed under the MIT License.
