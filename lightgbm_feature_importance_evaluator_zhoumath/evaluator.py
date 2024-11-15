# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:32:07 2024

@author: zhoushus
"""

import logging
import pandas as pd
import numpy as np
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import shap
import warnings
from .utils import preprocess_data, filter_features

warnings.filterwarnings("ignore", category=UserWarning)

class FeatureImportanceEvaluator:
    def __init__(self, data: pd.DataFrame, target_column: str, timestamp_column: str, feature_columns: list, model, 
                 model_params: dict = None, importance_method: str = 'gain', n_splits: int = 5, 
                 random_seed: int = 42, importance_threshold: float = 0, missing_values: list = None, 
                 repeats: int = 3, start_date: str = None, end_date: str = None, early_stopping_rounds: int = 100,
                 num_boost_round: int = 1000, verbose: bool = False, permutation_repeats: int = 5, n_jobs: int = -1):
        """
        Initialize the FeatureImportanceEvaluator.
        """
        
        self.data = data
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.feature_columns = feature_columns
        self.model = model
        self.model_params = model_params or {}
        self.importance_method = importance_method
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.importance_threshold = importance_threshold
        self.missing_values = missing_values or []
        self.repeats = repeats
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose = verbose
        self.permutation_repeats = permutation_repeats
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        self.model_params.update({
            'n_jobs': self.n_jobs,
            'force_col_wise': True,
            'n_estimators': self.num_boost_round
        })
        self.features, self.target = preprocess_data(
            data, target_column, timestamp_column, feature_columns, 
            start_date, end_date, missing_values
        )
        self.importance_methods = {
            'gain': self._gain_importance,
            'split': self._split_importance,
            'permutation': self._permutation_importance,
            'shap': self._shap_importance,
            'abs_shap': self._abs_shap_importance,
            'drop_column': self._drop_column_importance,
        }

    def _preprocess_data(self):
        """
        Preprocess data: filter by date, handle missing values, and optimize memory usage.
        """
        
        data = self.data.copy()
        data[self.timestamp_column] = pd.to_datetime(data[self.timestamp_column], errors='coerce')

        if self.start_date:
            data = data[data[self.timestamp_column] >= self.start_date]
            
        if self.end_date:
            data = data[data[self.timestamp_column] <= self.end_date]

        data.replace(self.missing_values, np.nan, inplace=True)
        features = data[self.feature_columns]
        target = data[self.target_column]
        
        return features, target

    def evaluate_importance(self):
        """
        Evaluate feature importance across multiple stratified folds.
        """
        
        all_scores = []
    
        for repeat in range(self.repeats):
            print(f"\nRepeat {repeat + 1}/{self.repeats}")
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed + repeat)
            fold_scores = []
    
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.features, self.target), 1):
                print(f"  Fold {fold}/{self.n_splits}")
                X_train, X_val = self.features.iloc[train_idx].reset_index(drop=True), self.features.iloc[val_idx].reset_index(drop=True)
                y_train, y_val = self.target.iloc[train_idx].reset_index(drop=True), self.target.iloc[val_idx].reset_index(drop=True)
                
                self.model.set_params(**self.model_params)
                self.model.fit(
                    X_train, 
                    y_train, 
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=[
                        early_stopping(self.early_stopping_rounds),
                        log_evaluation(20 if self.verbose else 0)
                    ]
                )
                importance_function = self.importance_methods.get(self.importance_method)
                
                if not importance_function:
                    raise ValueError(f"Invalid importance method: {self.importance_method}")
                    
                fold_importance = importance_function(X_train, X_val, y_val)
                fold_scores.append(fold_importance)
    
            repeat_importance = pd.concat(fold_scores, axis=1).mean(axis=1).to_frame('Importance').reset_index()
            repeat_importance.columns = ['Feature', 'Importance']
            repeat_importance.set_index('Feature', inplace=True)
            all_scores.append(repeat_importance)
    
        final_importance = pd.concat(all_scores, axis=1).mean(axis=1).to_frame('Importance').reset_index()
        final_importance.columns = ['Feature', 'Importance']
        final_importance.set_index('Feature', inplace=True)
        
        return final_importance

    def _gain_importance(self, X_train, X_val, y_val) -> pd.DataFrame:
        """
        Compute feature importance using the model's gain-based importance.
        """
        
        result = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.booster_.feature_importance(importance_type='gain')
        }).set_index('Feature')
        
        return result

    def _split_importance(self, X_train, X_val, y_val) -> pd.DataFrame:
        """
        Compute feature importance using the model's split-based importance.
        """
        
        result = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.booster_.feature_importance(importance_type='split')
        }).set_index('Feature')
        
        return result

    def _permutation_importance(self, X_train, X_val, y_val) -> pd.DataFrame:
        """
        Compute feature importance using permutation importance.
        """
        
        perm_importance = permutation_importance(
            self.model, 
            X_val, 
            y_val, 
            scoring='roc_auc',
            n_repeats=self.permutation_repeats, 
            random_state=self.random_seed, 
            n_jobs=self.n_jobs
        )
        result = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': perm_importance.importances_mean
        }).set_index('Feature')
        
        return result

    def _shap_importance(self, X_train, X_val, y_val) -> pd.DataFrame:
        """
        Parallelized computation of SHAP importance with improved efficiency.
        """
       
        explainer = shap.TreeExplainer(self.model.booster_)
        shap_values = explainer.shap_values(X_val)
        shap_importance = np.mean(shap_values, axis=0)
        result = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': shap_importance
        }).set_index('Feature')
        
        return result
    
    def _abs_shap_importance(self, X_train, X_val, y_val) -> pd.DataFrame:
        """
        Parallelized computation of SHAP importance with improved efficiency.
        """
       
        explainer = shap.TreeExplainer(self.model.booster_)
        shap_values = explainer.shap_values(X_val)
        shap_importance = np.mean(np.abs(shap_values), axis=0)
        result = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': shap_importance
        }).set_index('Feature')
        
        return result
    
    def _drop_column_importance(self, X_train, X_val, y_val) -> pd.DataFrame:
        """
        Compute feature importance using drop-column importance.
        """
        
        self.model.fit(
            X_train,
            self.target.iloc[X_train.index],
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                early_stopping(self.early_stopping_rounds),
                log_evaluation(20 if self.verbose else 0)
            ]
        )
        baseline_preds = self.model.predict_proba(X_val)[:, 1]
        baseline_score = roc_auc_score(y_val, baseline_preds)
        drop_column_importance = []
        
        for feature in self.feature_columns:
            X_train_dropped = X_train.drop(columns=[feature])
            X_val_dropped = X_val.drop(columns=[feature])
            dropped_model = self.model.__class__(**self.model_params)
            dropped_model.fit(
                X_train_dropped,
                self.target.iloc[X_train_dropped.index],
                eval_set=[(X_val_dropped, y_val)],
                eval_metric="auc",
                callbacks=[
                    early_stopping(self.early_stopping_rounds),
                    log_evaluation(20 if self.verbose else 0)
                ]
            )
            dropped_preds = dropped_model.predict_proba(X_val_dropped)[:, 1]
            dropped_score = roc_auc_score(y_val, dropped_preds)
            importance = baseline_score - dropped_score
            drop_column_importance.append(importance)
        
        result = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': drop_column_importance
        }).set_index('Feature')
        
        return result

    def get_filtered_features(self, importance_df):
        """
        Filter features based on the importance threshold using the utils function.
        """
        
        selected_features, sorted_importance = filter_features(importance_df, self.importance_threshold)
        print(f"Selected {len(selected_features)} features with importance above {self.importance_threshold}.")
        
        return selected_features, sorted_importance
    
    