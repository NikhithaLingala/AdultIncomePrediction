# Adult Income Prediction Using Machine Learning

A machine learning classification project that predicts whether an individual's annual income is **≤50K or >50K** using demographic, educational, employment and financial features.

The project follows an end-to-end ML workflow, including data cleaning, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning and classification threshold optimization.

## 📌 Project Highlights

| Metric | Result |
|---|---:|
| Dataset Size | 32,561 rows |
| Input Features | 13 |
| Best Model | XGBoost |
| Test Accuracy | 86.37% |
| Test Precision | 71.67% |
| Test Recall | 71.81% |
| Test F1 Score | 71.74% |
| ROC-AUC | 92.16% |
| Optimized Threshold | 0.69 |

## 📑 Table of Contents

- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Model Comparison](#model-comparison)
- [Final Model Evaluation](#final-model-evaluation)
- [Model Interpretation](#model-interpretation)
- [Project Structure](#project-structure)
- [Installation and Execution](#installation-and-execution)
- [Key Learnings](#key-learnings)
- [Author](#author)

## 🎯 Business Problem

Income classification can help understand how demographic, educational and employment-related factors relate to income levels.

The objective of this project is to develop a machine learning classification model that predicts whether an individual's annual income is:

- `<=50K`
- `>50K`

## 📊 Dataset

The project uses the Adult Income dataset.

**Dataset details:**

- Original rows: 32,561
- Original columns: 15
- Target variable: Income
- Input features after preprocessing: 13

**Target distribution:**

| Income Category | Distribution |
|---|---:|
| <=50K | 75.9% |
| >50K | 24.1% |

The dataset contains numerical and categorical features related to individuals' demographic and employment information.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Joblib
- Streamlit

## ⚙️ Project Workflow

### 1. Data Understanding
- Dataset inspection and dimensionality analysis
- Feature types and target distribution
- Missing-value analysis
- Duplicate detection
- Numerical and categorical feature analysis

### 2. Data Cleaning
- Replaced `?` values with missing values
- Handled missing categorical values using mode imputation
- Removed duplicate records
- Removed unnecessary Final_Weight column
- Standardized column names

### 3. Exploratory Data Analysis (EDA)
- Target distribution analysis
- Numerical feature distributions
- Categorical feature analysis
- Outlier detection
- Correlation analysis
- Data visualization using Matplotlib and Seaborn

### 4. Feature Engineering
- Log transformation of skewed features
- Created Has_Capital_Gain and Has_Capital_Loss features
- Grouped relevant categorical information
- Examined highly correlated numerical features

### 5. Train-Test Split

- Training data: 80%
- Testing data: 20%
- Stratified splitting
- Random state: 42

### 6. Feature Preprocessing

Used `ColumnTransformer` to apply different preprocessing techniques:

- StandardScaler for selected numerical features
- RobustScaler for skewed and outlier-sensitive numerical features
- OneHotEncoder for categorical features
- Passthrough for binary gender feature

### 7. Model Comparison

Four machine learning algorithms were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

### 8. Hyperparameter Tuning

Used `RandomizedSearchCV` with:

- 5-fold Stratified Cross-Validation
- 40 parameter combinations
- F1 Score as the optimization metric

Best recorded cross-validation F1 Score: **0.7177**

### 9. Threshold Optimization

The classification threshold was optimized using a validation split instead of relying only on the default 0.5 threshold.

Selected threshold: **0.69**

## 📈 Model Comparison

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|
| XGBoost | 83.28% | 70.88% | 92.25% |
| LightGBM | 83.08% | 70.38% | 92.12% |
| Logistic Regression | 80.02% | 66.60% | 89.45% |
| Random Forest | 78.93% | 66.55% | 90.29% |

XGBoost was selected for further hyperparameter tuning based on the recorded model comparison results.

## 🏆 Final Model Evaluation

The tuned XGBoost model was evaluated using the optimized threshold of 0.69.

**Final test results:**

| Metric | Score |
|---|---:|
| Accuracy | 86.37% |
| Precision | 71.67% |
| Recall | 71.81% |
| F1 Score | 71.74% |
| ROC-AUC | 92.16% |
| PR-AUC | 81.96% |

The notebook also includes:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Threshold Comparison
- Train-Test F1 Comparison
- Feature Importance Analysis

## 🔍 Model Interpretation

Feature importance was extracted from the final XGBoost model to identify influential features.

The notebook includes a visualization of the top 15 feature importances.

## 📁 Project Structure

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

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/NikhithaLingala/AdultIncomePrediction.git
```

### 2. Navigate to the project directory

```bash
cd AdultIncomePrediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `adult_salary_prediction_XGB.ipynb` to explore the complete machine learning workflow.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application uses the saved model pipeline and its associated threshold. Ensure the model file and required dependencies are available.

## 📚 Key Learnings

Through this project, I gained practical experience in:

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Handling imbalanced classification data
- Feature scaling and encoding
- Comparing machine learning algorithms
- Hyperparameter tuning
- Cross-validation
- Classification evaluation metrics
- Classification threshold optimization
- Model interpretation
- Saving trained models using Joblib
- Preparing ML applications for deployment

## 👩‍💻 Author

**Nikhitha Lingala**

B.Tech – Electronics and Communication Engineering (2025)

- GitHub: [NikhithaLingala](https://github.com/NikhithaLingala)
  

