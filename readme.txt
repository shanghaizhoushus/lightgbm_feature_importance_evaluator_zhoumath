# lightgbm_feature_importance_evaluator_zhoumath

A Python package for evaluating feature importance in LightGBM models using various methods. This package is tailored for LightGBM users but also supports model-agnostic feature importance evaluation methods, such as permutation and drop-column importance.

---

## **Key Features**

- **LightGBM-Specific Importance Methods**:
  - Gain-based importance.
  - Split-based importance.
- **Model-Agnostic Methods**:
  - Permutation importance.
  - Drop-column importance.
- **SHAP-Based Interpretability**:
  - Local and global explanations using SHAP values.
- **Cross-Validation Support**:
  - Robust feature importance evaluation using stratified k-fold cross-validation.
- **Feature Filtering**:
  - Select important features based on a user-defined threshold.

---

## **Installation**

You can install the package directly from PyPI:
pip install lightgbm_feature_importance_evaluator_zhoumath

Usage
Here’s how to use lightgbm_feature_importance_evaluator_zhoumath step by step:

1. Import the Package
from lightgbm_feature_importance_evaluator_zhoumath.evaluator import FeatureImportanceEvaluator
from lightgbm import LGBMClassifier
import pandas as pd

2. Prepare Your Dataset
# Sample dataset
data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [5, 4, 3, 2, 1],
    "feature3": [2, 3, 4, 5, 6],
    "target": [0, 1, 0, 1, 0],
    "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
})

3. Initialize the Evaluator
evaluator = FeatureImportanceEvaluator(
    data=data,
    target_column="target",
    timestamp_column="timestamp",
    feature_columns=["feature1", "feature2", "feature3"],
    model=LGBMClassifier(),
    importance_method="gain",  # Choose from 'gain', 'split', 'permutation', 'shap', 'drop_column'
)

4. Evaluate Feature Importance
importance_df = evaluator.evaluate_importance()
print("Feature Importance:")
print(importance_df)

5. Filter Features
selected_features, sorted_importance = evaluator.get_filtered_features(importance_df)
print("Selected Features:")
print(selected_features)

Available Importance Methods:
gain	#Importance based on the gain (performance improvement) when a feature is used for splitting.
split	#Importance based on the frequency a feature is used for splitting.
permutation	#Model-agnostic importance based on the drop in performance when a feature is randomly shuffled.
shap	#Uses SHAP values to explain the contribution of each feature to predictions.
drop_column	#Measures the change in model performance when a feature is entirely removed from the dataset.

Customization Options
Cross-Validation:
Adjust the number of splits using n_splits.
Repeat cross-validation multiple times using repeats.
Importance Threshold:
Filter features with importance above a specific threshold using importance_threshold.
Date Filtering:
Use start_date and end_date to filter data based on a time range.

Dependencies
The package requires the following Python libraries:
numpy
pandas
scikit-learn
shap
lightgbm
Install them with:
pip install -r requirements.txt

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

Acknowledgments
This package was developed to streamline the evaluation of feature importance in LightGBM models and simplify workflows for data scientists and machine learning practitioners.

Author
Zhoushus
Email: zhoushus@foxmail.com
GitHub: https://github.com/shanghaizhoushus