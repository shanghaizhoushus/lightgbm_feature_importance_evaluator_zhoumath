# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:34:04 2024

@author: 小米
"""

def print_summary(importance_df, threshold=0):
    """
    Print a summary of feature importance.

    Args:
        importance_df (pd.DataFrame): Feature importance dataframe.
        threshold (float): Minimum importance value to display.
    """
    filtered = importance_df[importance_df["Importance"] > threshold]
    print(f"Number of features above threshold ({threshold}): {len(filtered)}")
    print(filtered.sort_values(by="Importance", ascending=False))