# Customer Churn Prediction for Beta Bank

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-blueviolet)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Beta Bank faces the challenge of gradually losing customers month after month. This project develops a machine learning predictive model to identify customers with a high probability of leaving the bank (churn). The main objective is to create a classification model that maximizes the F1-score (with a minimum required value of 0.59) and complements the analysis with the AUC-ROC metric for comprehensive performance evaluation.

The project follows a structured methodology including data loading and exploration, preprocessing, class imbalance analysis, balancing techniques, parameter tuning, and final model evaluation.

## 🎯 Project Results
The project has successfully developed a machine learning model capable of accurately identifying customers with high probability of leaving the bank. After an exhaustive process, we achieved a model that exceeds the initial target:
* **F1-score**: 0.593 (exceeding the 0.59 target).
* **AUC-ROC**: 0.87
* **Model Precision**: 87% area under the ROC curve.

## 🚀 Business Impact
* Early identification of at-risk customers
* Proactive and personalized retention strategies.
* Resource optimization by focusing efforts on high-probability churn customers.
* Tangible reduction in churn rate and increased retention revenue.
* Improved customer satisfaction through preventive attention.

## 🎯 Core Skills
* **Data Preprocessing**: Handling missing values, categorical variable encoding, feature scaling.
* **Exploratory Analysis**: Class imbalance identification, distribution analysis.
* **Feature Engineering**: Data transformation and preparation for modeling.
* **Model Selection**: Comparison and evaluation of multiple algorithms.
* **Optimization**: Hyperparameter tuning and balancing techniques.
* **Model Evaluation**: Comprehensive analysis using multiple metrics.
* **Business Problem Solving**: Practical approach to addressing business challenges.

## 🛠️ Tech Stack
* **Machine Learning** → Scikit-learn
* **Backend** → Python 3.8+, Pandas, NumPy
* **Visualization** → Matplotlib, Seaborn
* **Development** → Jupyter Notebooks

## Local Execution
1. Clone the repository:

git clone https://github.com/RosellaAM/Megaline-Plan-Recommendation.git

2. Install dependencies:

pip install -r requirements.txt

3. Run analysis:

  jupyter notebook notebooks/prediccion_churn_beta_bank.ipynb
