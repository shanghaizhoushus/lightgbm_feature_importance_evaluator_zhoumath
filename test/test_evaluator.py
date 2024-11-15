# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:39:14 2024

@author: zhoushus
"""

import pytest
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm_feature_importance_evaluator_zhoumath.evaluator import FeatureImportanceEvaluator


@pytest.fixture
def sample_data():
    """Fixture to create a sample dataset."""
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "feature3": [2, 3, 4, 5, 6],
        "target": [0, 1, 0, 1, 0],
        "timestamp": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ],
    })
    return data


def test_initialization(sample_data):
    """Test the initialization of the evaluator."""
    evaluator = FeatureImportanceEvaluator(
        data=sample_data,
        target_column="target",
        timestamp_column="timestamp",
        feature_columns=["feature1", "feature2", "feature3"],
        model=LGBMClassifier(),
        importance_method="gain",
    )
    assert evaluator.target_column == "target"
    assert evaluator.timestamp_column == "timestamp"
    assert "feature1" in evaluator.feature_columns


def test_preprocessing(sample_data):
    """Test data preprocessing logic."""
    evaluator = FeatureImportanceEvaluator(
        data=sample_data,
        target_column="target",
        timestamp_column="timestamp",
        feature_columns=["feature1", "feature2", "feature3"],
        model=LGBMClassifier(),
        importance_method="gain",
    )
    X, y = evaluator._preprocess_data()
    assert len(X) == len(y)
    assert X.shape[1] == 3  # 3 features
    assert y.name == "target"


def test_evaluate_importance_gain(sample_data):
    """Test feature importance evaluation with gain method."""
    evaluator = FeatureImportanceEvaluator(
        data=sample_data,
        target_column="target",
        timestamp_column="timestamp",
        feature_columns=["feature1", "feature2", "feature3"],
        model=LGBMClassifier(),
        importance_method="gain",
    )
    importance_df = evaluator.evaluate_importance()
    assert not importance_df.empty
    assert set(importance_df.index) == {"feature1", "feature2", "feature3"}


def test_evaluate_importance_permutation(sample_data):
    """Test feature importance evaluation with permutation method."""
    evaluator = FeatureImportanceEvaluator(
        data=sample_data,
        target_column="target",
        timestamp_column="timestamp",
        feature_columns=["feature1", "feature2", "feature3"],
        model=LGBMClassifier(),
        importance_method="permutation",
    )
    importance_df = evaluator.evaluate_importance()
    assert not importance_df.empty
    assert set(importance_df.index) == {"feature1", "feature2", "feature3"}


def test_filter_features(sample_data):
    """Test filtering features based on importance threshold."""
    evaluator = FeatureImportanceEvaluator(
        data=sample_data,
        target_column="target",
        timestamp_column="timestamp",
        feature_columns=["feature1", "feature2", "feature3"],
        model=LGBMClassifier(),
        importance_method="gain",
    )
    importance_df = pd.DataFrame({
        "Importance": [0.1, 0.05, 0.2],
    }, index=["feature1", "feature2", "feature3"])
    selected_features, sorted_importance = evaluator.get_filtered_features(importance_df)
    assert "feature1" in selected_features
    assert "feature2" not in selected_features  # Assuming threshold > 0.05
