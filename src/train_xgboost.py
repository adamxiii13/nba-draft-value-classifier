"""
NBA Draft Value Classifier - XGBoost Training Pipeline
======================================================
This script trains an Extreme Gradient Boosting (XGBoost) classifier to project 
the NBA success of college prospects. It merges traditional NCAA box score stats 
with physical combine measurements, high school recruiting pedigree (RSCI), 
and draft age to identify historical Busts, Steals, Stars, and Superstars.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Define Directory Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'

def train_xgboost():
    """
    Executes the full machine learning pipeline:
    1. Ingests raw data and merges RSCI/Age context.
    2. Imputes missing data and applies synthetic oversampling (SMOTE) to balance rare classes (e.g., Superstars).
    3. Trains the XGBoost classifier and evaluates performance.
    4. Saves the model, label encoder, and feature columns for future inference.
    """
    print("🚀 Loading the final ML dataset for XGBoost...")
    dataset_path = os.path.join(DATA_DIR, 'final_ml_dataset.csv')
    df = pd.read_csv(dataset_path).dropna(subset=['ncaa_per'])

    print("🧬 Merging RSCI Pedigree and Draft Ages...")
    # Merge High School Pedigree (RSCI)
    rsci_path = os.path.join(DATA_DIR, 'historical_rsci.csv')
    rsci_df = pd.read_csv(rsci_path).drop_duplicates(subset=['player_name'])
    df = pd.merge(df, rsci_df[['player_name', 'rsci_rank']], on='player_name', how='left')
    
    # THE UNRANKED PENALTY: If a player was not a top 100 recruit, assign a baseline rank of 150
    df['rsci_rank'] = df['rsci_rank'].fillna(150) 

    # Merge Draft Ages
    age_path = os.path.join(DATA_DIR, 'historical_draft_ages.csv')
    age_df = pd.read_csv(age_path).drop_duplicates(subset=['player_name'])
    df = pd.merge(df, age_df, on='player_name', how='left')
    
    # If a player wasn't caught in the scrape, impute an average draft age of 21.0
    df['draft_age'] = df['draft_age'].fillna(21.0) 

    # Clean Feature List (Strictly raw metrics to prevent out-of-distribution overfitting)
    features = [
        'overall_pick', 'ncaa_stocks', 'ncaa_ast_tov_ratio', 
        'ncaa_per', 'ncaa_ts_pct', 'ncaa_bpm', 'ncaa_usg_pct', 'ncaa_class',
        'height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age'
    ]
    
    # One-Hot Encode the categorical NCAA Class variable
    X = pd.get_dummies(df[features], columns=['ncaa_class'])
    
    # Encode target labels (Bust, Expected, Steal, Star, Superstar) into integers
    le = LabelEncoder()
    y = le.fit_transform(df['draft_value_label'])

    print("🛠️ Imputing missing values and splitting data...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    print("⚖️ Balancing classes with SMOTE...")
    # Generate synthetic samples for rare categories (like Superstars) to prevent model bias
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("🧠 Training Extreme Gradient Boosting (XGBoost) Engine...")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # Evaluate Model
    y_pred = model.predict(X_test)
    print("\n--- FINAL XGBOOST PERFORMANCE ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Output Feature Importance
    print("\n📊 FEATURE IMPORTANCE (Top 10 Metrics):")
    importances = model.feature_importances_
    feature_names = X.columns
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
    print(fi_df.to_string(index=False))

    print("\n💾 Saving XGBoost model and Label Encoder...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(model, os.path.join(MODEL_DIR, 'nba_xgboost_model.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl')) 
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'xgboost_columns.pkl'))
    
    print(f"✅ XGBoost model successfully saved to {MODEL_DIR}/")

if __name__ == "__main__":
    train_xgboost()