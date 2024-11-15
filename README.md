
# LightGBM Feature Importance Evaluator

LightGBM Feature Importance Evaluator provides advanced tools to analyze and evaluate feature importance in LightGBM models using various methods, including traditional gains, SHAP values, and more. The package also includes enhanced preprocessing and feature filtering utilities.

---

## Key Features in v0.1.1

- **New Importance Method**: `abs_shap` - Computes mean absolute SHAP values for improved insights.
- **Enhanced Preprocessing**: Streamlined data filtering by date and handling of missing values via `preprocess_data`.
- **Feature Selection Utility**: Quickly filter features using `filter_features` based on importance thresholds.
- **Improved Performance**: Optimized functions for better runtime efficiency and usability.
- **Additional Parameters**: Support for verbosity and early stopping in LightGBM model training.

## Key Features

- **LightGBM-Specific Importance Methods**
  - Gain-based importance
  - Split-based importance

- **Model-Agnostic Methods**
  - Permutation importance
  - Drop-column importance

- **SHAP-Based Interpretability**
  - Local and global explanations using SHAP values

- **Cross-Validation Support**
  - Robust feature importance evaluation using stratified k-fold cross-validation

- **Feature Filtering**
  - Select important features based on a user-defined threshold

---

## Installation

Install the package directly from PyPI:

```sh
pip install lightgbm_feature_importance_evaluator_zhoumath
```

## Usage

Here's how to use `lightgbm_feature_importance_evaluator_zhoumath` step by step:

### 1. Import the Package

```python
from lightgbm_feature_importance_evaluator_zhoumath.evaluator import FeatureImportanceEvaluator
from lightgbm import LGBMClassifier
import pandas as pd
```

### 2. Prepare Your Dataset

```python
# Sample dataset
data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [5, 4, 3, 2, 1],
    "feature3": [2, 3, 4, 5, 6],
    "target": [0, 1, 0, 1, 0],
    "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
})
```

### 3. Initialize the Evaluator

```python
evaluator = FeatureImportanceEvaluator(
    data=data,
    target_column="target",
    timestamp_column="timestamp",
    feature_columns=feature_columns,
    model=model,
    importance_method="abs_shap",  # Use the new `abs_shap` method
    start_date="2023-01-01",       # Optional: Filter by start date
    end_date="2023-12-31",         # Optional: Filter by end date
    missing_values=["NA", None],   # Handle missing values
    verbose=True,                  # Print logs during processing
    num_boost_round=1000,          # Number of boosting rounds
    early_stopping_rounds=50,      # Early stopping patience
)
```

### 4. Evaluate Feature Importance

```python
importance_df = evaluator.evaluate_importance()
print("Feature Importance:")
print(importance_df)
```

### 5. Filter Features

```python
selected_features, sorted_importance = evaluator.get_filtered_features(importance_df)
print("Selected Features:")
print(selected_features)
```

## Available Importance Methods

- **gain**: Importance based on the gain (performance improvement) when a feature is used for splitting.
- **split**: Importance based on the frequency a feature is used for splitting.
- **permutation**: Model-agnostic importance based on the drop in performance when a feature is randomly shuffled.
- **shap**: Uses SHAP values to explain the contribution of each feature to predictions.
- **drop_column**: Measures the change in model performance when a feature is entirely removed from the dataset.

---

## Customization Options

### Cross-Validation
- Adjust the number of splits using `n_splits`.
- Repeat cross-validation multiple times using `repeats`.

### Importance Threshold
- Filter features with importance above a specific threshold using `importance_threshold`.

### Date Filtering
- Use `start_date` and `end_date` to filter data based on a time range.

## Dependencies

The package requires the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `shap`
- `lightgbm`

Install them with:

```sh
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## Acknowledgments

This package was developed to streamline the evaluation of feature importance in LightGBM models and simplify workflows for data scientists and machine learning practitioners.

## Author

- **Zhoushus**
  - Email: [zhoushus@foxmail.com](mailto:zhoushus@foxmail.com)
  - GitHub: [https://github.com/shanghaizhoushus](https://github.com/shanghaizhoushus)
