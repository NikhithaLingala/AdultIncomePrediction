# Adult Income Prediction Using Machine Learning

## Project Overview

This project predicts whether an individual's annual income is **<=50K or >50K** using demographic, employment, education, and financial-related features from the Adult Income dataset.

The project follows an end-to-end machine learning workflow:

**Data Understanding → Data Cleaning → EDA → Feature Engineering → Preprocessing → Model Comparison → Hyperparameter Tuning → Threshold Optimization → Evaluation → Model Saving / Deployment Readiness**

## Dataset

- **Rows:** 32,561
- **Original columns:** 15
- **Target variable:** `Income`
- **Target classes:** `<=50K` and `>50K`

After preprocessing and removing unnecessary information, the modeling data contains 13 input features.

The target distribution is imbalanced:

- `<=50K`: 75.9%
- `>50K`: 24.1%

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Joblib
- Streamlit (application file included in the repository)

## Project Workflow

### 1. Data Understanding

The dataset was loaded and inspected to understand:

- Dataset dimensions
- Feature types
- Target distribution
- Missing values
- Duplicate records
- Numerical feature distributions
- Categorical feature distributions
- Outliers
- Correlations between numerical features

### 2. Data Cleaning

The project includes:

- Replacing `?` values with missing values (`NaN`)
- Checking missing-value percentages
- Handling missing categorical values using mode imputation
- Detecting and removing duplicate records
- Removing the `Final_Weight` column
- Standardizing column names
- Renaming columns to clearer names

### 3. Exploratory Data Analysis

EDA was performed to understand:

- Target-variable distribution
- Numerical feature distributions
- Categorical feature distributions
- Outliers in numerical variables
- Correlations among numerical variables

Visualizations include distribution plots, boxplots, and a correlation heatmap.

### 4. Feature Engineering

The project includes feature engineering based on the dataset's characteristics, including:

- Log transformation for skewed features such as `Capital_Gain` and `Capital_Loss`
- Creation of `Has_Capital_Gain` and `Has_Capital_Loss`
- Creation/grouping of relevant categorical information
- Handling of highly correlated numerical features

### 5. Train-Test Split

The dataset was divided using:

- **80% training data**
- **20% testing data**
- Stratified split to maintain the target-class distribution
- `random_state=42` for reproducibility

The resulting split was:

- Training: 26,029 rows
- Testing: 6,508 rows

### 6. Feature Preprocessing

A `ColumnTransformer` was used to apply different preprocessing methods to different feature groups.

- `StandardScaler` for selected numerical features
- `RobustScaler` for skewed/outlier-sensitive numerical features
- `OneHotEncoder` for categorical features
- Binary gender feature passed through directly

### 7. Model Comparison

Four classification models were trained and evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

Because the target classes are imbalanced, class-balancing approaches were used during model training.

### Model Comparison Results

| Model | Accuracy | F1 Score | ROC-AUC | PR-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.8328 | 0.7088 | 0.9225 | 0.8230 | 0.6107 | 0.8444 |
| LightGBM | 0.8308 | 0.7038 | 0.9212 | 0.8204 | 0.6087 | 0.8342 |
| Logistic Regression | 0.8002 | 0.6660 | 0.8945 | 0.7360 | 0.5577 | 0.8265 |
| Random Forest | 0.7893 | 0.6655 | 0.9029 | 0.7712 | 0.5389 | 0.8699 |

Based on the recorded model-comparison results, XGBoost was selected for further tuning.

## 8. XGBoost Hyperparameter Tuning

RandomizedSearchCV was used with:

- 5-fold Stratified Cross-Validation
- 40 parameter combinations
- F1 score as the optimization metric

The recorded best cross-validation F1 score was:

**CV F1 Score: 0.7177**

Selected parameters included:

- `n_estimators`: 400
- `max_depth`: 4
- `learning_rate`: 0.08
- `subsample`: 0.8
- `colsample_bytree`: 0.7
- `gamma`: 1.0
- `min_child_weight`: 5
- `reg_alpha`: 2.0
- `reg_lambda`: 3.0

## 9. Threshold Optimization

Instead of using only the default probability threshold of 0.5, the project evaluated different thresholds on a validation split.

The selected optimized threshold recorded in the notebook was:

**0.69**

At this threshold, the test-set results were:

- Accuracy: **0.8637**
- Precision: **0.7167**
- Recall: **0.7181**
- F1 Score: **0.7174**
- ROC-AUC: **0.9216**
- PR-AUC: **0.8196**

The ROC-AUC and PR-AUC are probability-based metrics, so changing the classification threshold does not change these values.

## 10. Final Model Evaluation

The final classification report at the optimized threshold showed:

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| <=50K | 0.91 | 0.91 | 0.91 |
| >50K | 0.72 | 0.72 | 0.72 |

Overall accuracy was approximately **86.37%**.

The notebook also includes:

- Confusion matrix
- ROC curve
- Precision-Recall curve
- Threshold comparison
- Train/test F1 comparison
- Feature-importance analysis

## 11. Model Interpretation

Feature importance was extracted from the final XGBoost model to identify the most influential features used by the model.

The notebook visualizes the top 15 feature importances.

## 12. Model Saving and Deployment Readiness

The final trained model and related metadata were saved using Joblib as:

`xgb_pipeline_v2.pkl`

The saved bundle includes:

- Trained preprocessing/model pipeline
- Classification threshold
- Model metadata
- Cross-validation F1 score
- Test F1 score
- PR-AUC
- Cross-validation gap
- Best hyperparameters
- Python and library version information

The repository also contains an `app.py` file and dependency/runtime files for the application.

## Project Structure

```text
AdultIncomePrediction/
│
├── adult.csv
├── adult_salary_prediction_XGB.ipynb
├── app.py
├── requirements.txt
├── runtime.txt
├── xgb_pipeline_v2.pkl
└── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/NikhithaLingala/AdultIncomePrediction.git
cd AdultIncomePrediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Explore the notebook

Open:

```text
adult_salary_prediction_XGB.ipynb
```

Run the notebook cells to reproduce the data analysis, preprocessing, model training, tuning, and evaluation workflow.

### 4. Run the application

The repository includes `app.py` for the application/deployment component. Use the command and setup specified by the application's dependencies.

## Key Learning Outcomes

Through this project, I worked on:

- Data cleaning and preprocessing
- Exploratory data analysis
- Handling missing values and duplicates
- Feature engineering
- Handling class imbalance
- Feature scaling and categorical encoding
- Comparing multiple machine learning models
- Hyperparameter tuning
- Cross-validation
- Classification metrics
- Threshold optimization
- Model interpretation
- Saving a trained ML model
- Preparing a model for application deployment

## Author

**Nikhitha Lingala**

GitHub: https://github.com/NikhithaLingala
