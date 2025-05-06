
import os
import json
import pathlib
from io import StringIO
import argparse
import pandas as pd
import numpy as np
import boto3
import sagemaker
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def model_fn(model_dir):
    """
    Load the model from the model_dir
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":

    print("[INFO] Extracting arguments")

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)


    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train_set.csv")
    parser.add_argument("--test-file", type=str, default="test_set.csv")
    args, _ = parser.parse_known_args()

    print("SKLearn version: ", sklearn.__version__)
    print("Joblib version: ", joblib.__version__) 

    print("[INFO] Loading the data")

    # Load the training data
    train_data = pd.read_csv(os.path.join(args.train, args.train_file))
    test_data = pd.read_csv(os.path.join(args.test, args.test_file))

    features = train_data.drop(columns=["isFraud"]).columns.tolist()
    target = train_data.columns.tolist().pop(train_data.columns.tolist().index("isFraud"))

    print("[INFO] Data loaded")
    print(f"Features: {features}")
    print(f"Target: {target}")
    print()
    print(f"Train data shape: {train_data.shape}")
    print()
    print(f"Test data shape: {test_data.shape}")
    print()

    # Split the data into features and labels
    print("[INFO] Training set, validation set and test set")
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print()


    # Building the model
    print("[INFO] Training the model")


    # Create the model
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)

    # Fit the model
    model.fit(X_train, y_train)
    print("[INFO] Model trained")
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    print(f"Model path: {model_path}")
    # Save the model
    joblib.dump(model, model_path)
    print("[INFO] Model saved")
    print(f"Model path: {model_path}")
    print()

    # Evaluate the model
    print("[INFO] Evaluating the model")
    # Make predictions
    y_pred_test = model.predict(X_test)
    #y_pred_proba_val = model.predict_proba(X_test)[:, 1]
    # Calculate the metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print()
    print("[INFO] Test metrics")
    print("Test rows: ", X_test.shape[0])
    print(f"Test accuracy: {test_accuracy}")
    print("Test classification report:")
    print(test_rep)
    print()
