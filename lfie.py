# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:09:30 2024

@author: 小米
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lightgbm_feature_importance_evaluator_zhoumath.evaluator import FeatureImportanceEvaluator
from lightgbm import LGBMClassifier
import pandas as pd
import os
from joblib import parallel_backend
import warnings

warnings.filterwarnings("ignore")

os.environ["JOBLIB_TEMP_FOLDER"] = "D:\\temp"


# Load the breast cancer dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Add a timestamp column for demonstration purposes
X['timestamp'] = pd.date_range(start='2023-01-01', periods=X.shape[0])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine X and y into a single DataFrame for the evaluator
train_data = X_train.copy()
train_data['target'] = y_train

model = LGBMClassifier(random_state=42)
feature_columns = X_train.columns[:-1].tolist()

# Define the evaluator
evaluator = FeatureImportanceEvaluator(
    data=train_data,
    target_column="target",
    timestamp_column="timestamp",
    feature_columns=feature_columns,
    model=model,
    model_params={},
    importance_method="drop_column",
    num_boost_round = 8964
)

# Evaluate feature importance
with parallel_backend('loky', n_jobs=-1):
    importance_df = evaluator.evaluate_importance()


print("\nFeature Importance (Gain-based):")
print(importance_df)

# Filter features with importance above a threshold
selected_features, sorted_importance = evaluator.get_filtered_features(importance_df)
print("\nSelected Features:")
print(selected_features)