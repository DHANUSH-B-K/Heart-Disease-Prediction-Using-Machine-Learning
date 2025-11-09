import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# 1. Load Data
df = pd.read_csv('cardio_train.csv', sep=';')

# 2. Feature Engineering
df = df.drop('id', axis=1)

# Add BMI
df['BMI'] = df['weight'] / ((df['height']/100) ** 2)

# Blood Pressure Features
df['ap_diff'] = df['ap_hi'] - df['ap_lo']
df['ap_hi_over_lo'] = df['ap_hi'] / (df['ap_lo'] + 1e-9)
df['is_hyper'] = ((df['ap_hi'] >= 140) & (df['ap_lo'] >= 90)).astype(int)

# Age in years
df['age_years'] = (df['age'] / 365).astype(int)
df['age_bucket'] = pd.cut(df['age_years'], bins=[0,30,40,50,60,100], labels=[0,1,2,3,4]).astype(int)


# Cholesterol & glucose as categorical
df['cholesterol'] = df['cholesterol'].replace({1:0, 2:1, 3:2})
df['gluc'] = df['gluc'].replace({1:0, 2:1, 3:2})
df['gender'] = df['gender'].map({1:0, 2:1})

# Smoking/Alcohol/Active interactions
df['bad_habits'] = df[['smoke', 'alco']].sum(axis=1)
df['active_bin'] = df['active'].astype(int)
df['smoke_alco'] = (df['smoke'] & df['alco']).astype(int)

# Drop original 'age'
df = df.drop('age', axis=1)

# Prepare X and y
X = df.drop('cardio', axis=1)
y = df['cardio']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Pipelines and Hyperparameter Tuning
# Standardize continuous variables only
cont_features = ['height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 'ap_diff', 'ap_hi_over_lo', 'age_years']

scaler = StandardScaler()
X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
X_test[cont_features] = scaler.transform(X_test[cont_features])

# Random Forest Hyperparameter Tuning
rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [6, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# XGBoost Hyperparameter Tuning
xgb_params = {
    'n_estimators': [100, 300],
    'max_depth': [5, 8, 12],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 1],
    'eval_metric': ['logloss']
}
xgb = XGBClassifier(random_state=42)
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

# 5. Evaluation Helper
def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n----- {model_name} Evaluation -----")
    print(f"Best Parameters: {getattr(model, 'get_params', lambda: {})()}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return acc, auc

# 6. Evaluate Both Models
rf_acc, rf_auc = evaluate(best_rf, X_test, y_test, "Random Forest")
xgb_acc, xgb_auc = evaluate(best_xgb, X_test, y_test, "XGBoost")

if xgb_auc > rf_auc:
    print("\nOverall, XGBoost performed BEST (higher ROC AUC score).")
else:
    print("\nOverall, Random Forest performed BEST (higher ROC AUC score).")