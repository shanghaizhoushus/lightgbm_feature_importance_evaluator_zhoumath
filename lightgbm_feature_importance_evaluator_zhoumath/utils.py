# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:32:20 2024

@author: zhoushus
"""

import pandas as pd
import numpy as np

def preprocess_data(data, target_column, timestamp_column, feature_columns, start_date=None, end_date=None, missing_values=None):
    """
    Preprocess data: filter by date, handle missing values, and optimize memory usage.
    """
    
    data = data.copy()
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors='coerce')

    if start_date:
        data = data[data[timestamp_column] >= pd.to_datetime(start_date)]
        
    if end_date:
        data = data[data[timestamp_column] <= pd.to_datetime(end_date)]

    data.replace(missing_values or [], np.nan, inplace=True)
    features = data[feature_columns]
    target = data[target_column]
    
    return features, target

def filter_features(importance_df, importance_threshold=0):
    """
    Filter features based on the importance threshold.
    """
    
    sorted_importance = importance_df.sort_values(by='Importance', ascending=False)
    selected_features = sorted_importance[sorted_importance['Importance'] > importance_threshold].index.tolist()
    
    return selected_features, sorted_importance
