import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# 1. Data Preprocessing and Feature Engineering (same as your training setup)
def preprocess_input(df):
    # Add BMI
    df['BMI'] = df['weight'] / ((df['height']/100) ** 2)
    # Blood Pressure Features
    df['ap_diff'] = df['ap_hi'] - df['ap_lo']
    df['ap_hi_over_lo'] = df['ap_hi'] / (df['ap_lo'] + 1e-9)
    df['is_hyper'] = ((df['ap_hi'] >= 140) & (df['ap_lo'] >= 90)).astype(int)
    # Age in years and age_bucket
    df['age_years'] = (df['age'] / 365).astype(int)
    df['age_bucket'] = pd.cut(df['age_years'], bins=[0,30,40,50,60,100], labels=[0,1,2,3,4]).astype(int)
    # Cholesterol, glucose, gender
    df['cholesterol'] = df['cholesterol'].replace({1:0, 2:1, 3:2}).astype(int)
    df['gluc'] = df['gluc'].replace({1:0, 2:1, 3:2}).astype(int)
    df['gender'] = df['gender'].map({1:0, 2:1}).astype(int)
    # Smoking/Alcohol/Active interactions
    df['bad_habits'] = df[['smoke', 'alco']].sum(axis=1)
    df['active_bin'] = df['active'].astype(int)
    df['smoke_alco'] = (df['smoke'] & df['alco']).astype(int)
    # Remove original 'age' if present
    if 'age' in df.columns:
        df = df.drop('age', axis=1)
    # Keep columns as in model training (without 'cardio')
    return df

# 2. Training code (run once for model and scaler creation)
def train_and_save_model():
    # Load data
    df = pd.read_csv('cardio_train.csv', sep=';')
    df = df.drop('id', axis=1)
    y = df['cardio']
    X = preprocess_input(df.drop('cardio', axis=1))

    # Standardize
    cont_features = ['height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 'ap_diff',
                     'ap_hi_over_lo', 'age_years']
    scaler = StandardScaler()
    X[cont_features] = scaler.fit_transform(X[cont_features])

    # Use the best found parameters for XGBoost
    xgb_best = XGBClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    xgb_best.fit(X, y)

    # Save model and scaler
    joblib.dump(xgb_best, 'xgb_cardio_model.joblib')
    joblib.dump(scaler, 'cardio_scaler.joblib')
    joblib.dump(list(X.columns), 'cardio_features.joblib')
    print("Model and scaler saved.")

# Uncomment and run once to train and save
# train_and_save_model()

# 3. Prediction code for user input
def predict_user_input(user_input_dict):
    # Load model and scaler
    xgb_model = joblib.load('xgb_cardio_model.joblib')
    scaler = joblib.load('cardio_scaler.joblib')
    cols = joblib.load('cardio_features.joblib')

    # Prepare user input as DataFrame (must match original feature names)
    df_input = pd.DataFrame([user_input_dict])
    df_input = preprocess_input(df_input)

    # Reindex so columns are in model order (add missing if needed)
    df_input = df_input.reindex(cols, axis=1)
    cont_features = ['height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 'ap_diff',
                     'ap_hi_over_lo', 'age_years']
    df_input[cont_features] = scaler.transform(df_input[cont_features])

    # Make prediction
    pred_prob = xgb_model.predict_proba(df_input)[0, 1]
    pred_label = xgb_model.predict(df_input)[0]
    print(f"Heart disease predicted (cardio=1): {bool(pred_label)}, Probability: {pred_prob:.3f}")
def print_inputs(input_dict):
    print("Here are the preset example input values to be used:\n")
    for k,v in input_dict.items():
        print(f"{k}: {v} ({descriptions[k]})")
def get_custom_input():
    custom_input = {}
    for k in example_input.keys():
        prompt = f"Enter value for {k} [{descriptions[k]}]: "
        val = input(prompt)
        if val == '':
            val = example_input[k]  # Use example value as default on blank
        else:
            val = type(example_input[k])(val)  # Cast to correct type
        custom_input[k] = val
    return custom_input

# 4. Example usage
if __name__ == "__main__":
    # Example user input, must match training features (except 'cardio' and 'id')
    # Pre-filled example input
    example_input = {
    'age': 18250,       # age in days (e.g., 50 years)
    'gender': 2,        # 1: woman, 2: man
    'height': 170,
    'weight': 80,
    'ap_hi': 145,
    'ap_lo': 95,
    'cholesterol': 2,   # 1: normal, 2: above normal, 3: well above normal
    'gluc': 1,          # 1: normal, etc.
    'smoke': 0,
    'alco': 0,
    'active': 1
}

    descriptions = {
    'age': "Age in days (e.g., 18250 = 50 years)",
    'gender': "Gender (1: woman, 2: man)",
    'height': "Height in cm",
    'weight': "Weight in kg",
    'ap_hi': "Systolic blood pressure",
    'ap_lo': "Diastolic blood pressure",
    'cholesterol': "Cholesterol (1: normal, 2: above normal, 3: well above normal)",
    'gluc': "Glucose (1: normal, 2: above normal, 3: well above normal)",
    'smoke': "Smokes? (0: No, 1: Yes)",
    'alco': "Alcohol intake? (0: No, 1: Yes)",
    'active': "Physically active? (0: No, 1: Yes)"
}


# Main logic
#print_inputs(example_input)
user_confirm = input("\nType 'go ahead' to use these values, or anything else to enter your own: ").strip().lower()

if user_confirm == "go ahead":
    final_input = example_input
    print("\nUsing example input.")
else:
    print("\nOK, please enter your own values.")
    final_input = get_custom_input()
    print("\nYou entered:")
    print_inputs(final_input)

# Now, convert to DataFrame for feature engineering and prediction
sample = pd.DataFrame([final_input])
print("\nReady for prediction (dataframe):")
print(sample)
predict_user_input(final_input)