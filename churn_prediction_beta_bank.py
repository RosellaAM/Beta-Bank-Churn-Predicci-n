"""
Prediction of Customer Churn at Beta Bank with Machine Learning

This project develops a machine learning model to predict customer churn for Beta Bank.
The goal is to create a classification model that maximizes the F1 score (minimum required: 0.59)
and uses AUC-ROC for comprehensive performance evaluation.

Methodology:
1. Data loading and exploration
2. Data preprocessing
3. Class imbalance analysis
4. Balancing techniques
5. Parameter tuning
6. Final evaluation
"""

# Analysis and data visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data splitting and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Balancing techniques
from sklearn.utils import shuffle

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Load and examine the dataset
data = pd.read_csv('/datasets/Churn.csv')

print('General Information')
print(f'Dataset dimensions: {data.shape}')
print('\nColumn Information')
print(data.info())
print('\nSample of Data')
print(data.head())


# Data cleaning
# Check for null values
print('Null values:', data.isnull().sum())

# Check for duplicate values
print('\nDuplicate values:', data.duplicated().sum())

# Fix column names
data_clean = data.copy()
column_mapping = {
    'RowNumber': 'row_number',
    'CustomerId': 'customer_id', 
    'CreditScore': 'credit_score',
    'Geography': 'geography',
    'Gender': 'gender',
    'Age': 'age',
    'Tenure': 'tenure',
    'Balance': 'balance',
    'NumOfProducts': 'num_of_products',
    'HasCrCard': 'has_credit_card',
    'IsActiveMember': 'is_active_member',
    'EstimatedSalary': 'estimated_salary',
    'Exited': 'exited'
}
data_clean = data_clean.rename(columns=column_mapping)

# Remove unnecessary column
data_clean.drop(columns=['row_number'], inplace=True)

# Handle null values
numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if data_clean[col].isnull().sum() > 0:
        data_clean[col].fillna(data_clean[col].median())


# EDA
# Churn distribution
target_distribution = data_clean['exited'].value_counts()
plt.figure(figsize=(8, 8))
data_clean['exited'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Churn Distribution (Exited)')
plt.xlabel('Exited (0: No, 1: Yes)')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)

# Geography vs Churn
churn_rates = data_clean.groupby('geography')['exited'].agg(['count', 'mean']).reset_index()
churn_rates['churn_rate'] = churn_rates['mean'] * 100

plt.figure(figsize=(10, 6))
bars = plt.bar(churn_rates['geography'], churn_rates['churn_rate'], 
               color=['lightcoral' if x > 20 else 'lightblue' for x in churn_rates['churn_rate']])
plt.title('Churn Rate by Country')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Country')

# Correlation of numerical variables with churn
numeric_cols = ['age', 'credit_score', 'balance', 'estimated_salary', 'tenure']
correlation = []
for col in numeric_cols:
    corr = data_clean[col].corr(data_clean['exited'])
    correlation.append(corr)

plt.figure(figsize=(10, 6))
plt.bar(numeric_cols, correlation, color=['red' if x < 0 else 'green' for x in correlation])
plt.title('Correlation of Numerical Variables with Churn')
plt.ylabel('Correlation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Data preprocessing
# Encode labels for categorical variable handling
encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data_clean), columns=data_clean.columns)

# Split data into 60% training, 20% validation, and 20% test
features = data_ordinal.drop('exited', axis=1)
target = data_ordinal['exited']

# First split
features_train, features_temp, target_train, target_temp = train_test_split(
    features, target, 
    test_size=0.4, 
    random_state=42, 
    stratify=target
)
# Second split
features_valid, features_test, target_valid, target_test = train_test_split(
    features_temp, target_temp,
    test_size=0.5,
    random_state=42,
    stratify=target_temp
)

# Standardize data scales
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_valid_scaled = scaler.transform(features_valid)
features_test_scaled = scaler.transform(features_test)


# Class imbalance analysis
# Check class distribution in each set
print('Training Set Proportion')
print(target_train.value_counts(normalize=True))
print('\nValidation Set Proportion')
print(target_valid.value_counts(normalize=True))
print('\nTest Set Proportion')
print(target_test.value_counts(normalize=True))

# Train baseline model
model_baseline = DecisionTreeClassifier(random_state=42, max_depth=3)
model_baseline.fit(features_train_scaled, target_train)

# Predict on validation set
baseline_prediction = model_baseline.predict(features_valid_scaled)

# Evaluate metrics
print('Confusion Matrix:')
print(confusion_matrix(target_valid, baseline_prediction))
print('\nClassification Report:')
print(classification_report(target_valid, baseline_prediction, target_names=['No Churn', 'Churn']))


# Balancing techniques
# Oversampling function
def upsample(features, target, repeat):
    # Convert to DataFrame/Series if they are NumPy arrays
    features_df = pd.DataFrame(features).reset_index(drop=True)
    target_series = pd.Series(target).reset_index(drop=True)
        
    features_zeros = features_df[target_series == 0]
    features_ones = features_df[target_series == 1]
    target_zeros = target_series[target_series == 0]
    target_ones = target_series[target_series == 1]
    
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=42)
    
    return features_upsampled, target_upsampled

# Model evaluation function
def evaluate_model(model, features_val, target_val, model_name, balance_method):
    # Prediction
    t_prediction = model.predict(features_val)

    # Calculate metrics
    recall = recall_score(target_val, t_prediction, pos_label=1)
    precision = precision_score(target_val, t_prediction, pos_label=1)
    f1 = f1_score(target_val, t_prediction, pos_label=1)

    # Display results
    return {
        'Model:': model_name,
        'Technique: ': balance_method,
        'Recall Score: ': round(recall, 4),
        'Precision Score: ': round(precision, 4),
        'F1 Score: ': round(f1, 4)
    }

# Upsampled data
features_upsampled, target_upsampled = upsample(features_train_scaled, target_train, 10)

# DecisionTreeClassifier
    # Class weight adjustment
tree_balanced_1 = DecisionTreeClassifier(random_state=42, max_depth=5, class_weight='balanced')
tree_balanced_1.fit(features_train_scaled, target_train)

    # Upsampled model
tree_balanced_2 = DecisionTreeClassifier(random_state=42, max_depth=5)
tree_balanced_2.fit(features_upsampled, target_upsampled)

# RandomForestClassifier
    # Class weight adjustment
forest_balanced_1 = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced')
forest_balanced_1.fit(features_train_scaled, target_train)

    # Upsampled model
forest_balanced_2 = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
forest_balanced_2.fit(features_upsampled, target_upsampled)

# Confusion matrix
tree_1_pred = tree_balanced_1.predict(features_valid_scaled)
tree_2_pred = tree_balanced_2.predict(features_valid_scaled)
forest_1_pred = forest_balanced_1.predict(features_valid_scaled)
forest_2_pred = forest_balanced_2.predict(features_valid_scaled)

print('Confusion matrix decision tree model 1:\n', confusion_matrix(target_valid, tree_1_pred))
print('\nConfusion matrix decision tree model 2:\n', confusion_matrix(target_valid, tree_2_pred))
print('\nConfusion matrix random forest model 1:\n', confusion_matrix(target_valid, forest_1_pred))
print('\nConfusion matrix random forest model 2:\n', confusion_matrix(target_valid, forest_2_pred))

# Evaluate metrics
results = []
tree_1 = evaluate_model(
    tree_balanced_1, 
    features_valid_scaled, target_valid, 
    'Decision Tree Classifier', 'Class weight balanced'
)
tree_2 = evaluate_model(
    tree_balanced_2, 
    features_valid_scaled, target_valid, 
    'Decision Tree Classifier', 'Oversampling'
)
forest_1 = evaluate_model(
    forest_balanced_1, 
    features_valid_scaled, target_valid, 
    'Random Forest Classifier', 'Class weight balanced'
)
forest_2 = evaluate_model(
    forest_balanced_2, 
    features_valid_scaled, target_valid, 
    'Random Forest Classifier', 'Oversampling'
)

# Store results to add to a DataFrame
results.append(tree_1)
results.append(tree_2)
results.append(forest_1)
results.append(forest_2)

# Create results DataFrame
df_results = pd.DataFrame(results)
df_results


# Parameter tuning
# Tuning for decision tree
best_max_depth_tree = 0
best_f1_tree_score = 0

for depth in range(1, 11):
    d_tree = DecisionTreeClassifier(random_state=42, max_depth=depth, class_weight='balanced')
    d_tree.fit(features_train_scaled, target_train)

    d_tree_prediction = d_tree.predict(features_valid_scaled)
    d_tree_f1 = f1_score(target_valid, d_tree_prediction, pos_label=1)

    if d_tree_f1 > best_f1_tree_score:
        best_max_depth_tree = depth
        best_f1_tree_score = d_tree_f1

# Tuning for random forest
best_n_estimators = 0
best_max_depth_forest = 0
best_f1_forest_score = 0

for n_est in [50, 100, 150, 200]:
    for depth in range(1, 11):
        r_forest = RandomForestClassifier(random_state=42, n_estimators=n_est, max_depth=depth, class_weight='balanced')
        r_forest.fit(features_train_scaled, target_train)

        r_forest_prediction = r_forest.predict(features_valid_scaled)
        r_forest_f1 = f1_score(target_valid, r_forest_prediction, pos_label=1)

        if r_forest_f1 > best_f1_forest_score:
            best_n_estimators = n_est
            best_max_depth_forest = depth
            best_f1_forest_score = r_forest_f1

# Results
print("Model: Decision Tree")
print("Best max_depth:", best_max_depth_tree)
print("Best F1-Score:", round(best_f1_tree_score, 4))

print("\nModel: Random Forest")
print("Best n_estimators:", best_n_estimators)
print("Best max_depth:", best_max_depth_forest)
print("Best F1-Score:", round(best_f1_forest_score, 4))


# Final evaluation
# Final data
features_train_s = pd.DataFrame(features_train_scaled).reset_index(drop=True)
features_valid_s = pd.DataFrame(features_valid_scaled).reset_index(drop=True)
features_final = pd.concat([features_train_s, features_valid_s])
target_final = pd.concat([target_train, target_valid])

# Train model
final_model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=8, class_weight='balanced')
final_model.fit(features_final, target_final)

# Test prediction and probability
final_prediction = final_model.predict(features_test_scaled)
final_probabilities = final_model.predict_proba(features_test_scaled)[:, 1]

# Calculate metrics
final_f1 = f1_score(target_test, final_prediction, pos_label=1)
final_auc_roc_score = roc_auc_score(target_test, final_probabilities)

# Results
print('-Final Results-')
print('F1 Score:', round(final_f1, 4))
print('AUC-ROC Score:', round(final_auc_roc_score, 4))

# ROC curve
fpr, tpr, thresholds = roc_curve(target_test, final_probabilities)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
