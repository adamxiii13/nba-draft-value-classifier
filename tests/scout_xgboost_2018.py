import pandas as pd
import joblib
import numpy as np

def run_2018_backtest():
    print("🧠 Loading XGBoost Model & Translators...")
    model = joblib.load('models/nba_xgboost_model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    model_columns = joblib.load('models/xgboost_columns.pkl')

    print("📊 Loading 2018 Draft Class Data...")
    data = [
        {
            'player_name': 'Marvin Bagley III', 'overall_pick': 2, 'rsci_rank': 1,
            'height_in': 81.5, 'ape_index': 3.0, 'vertical_in': 38.0, 
            'ncaa_per': 30.6, 'ncaa_bpm': 10.0, 'ncaa_ast_tov_ratio': 0.99, 
            'ncaa_stocks': 1.8, 'ncaa_ts_pct': 0.640, 'ncaa_usg_pct': 25.0, 'ncaa_class': 'FR'
        },
        {
            'player_name': 'Trae Young', 'overall_pick': 5, 'rsci_rank': 23,
            'height_in': 72.5, 'ape_index': 0.5, 'vertical_in': 30.0, 
            'ncaa_per': 28.5, 'ncaa_bpm': 11.5, 'ncaa_ast_tov_ratio': 1.67, 
            'ncaa_stocks': 2.0, 'ncaa_ts_pct': 0.590, 'ncaa_usg_pct': 38.0, 'ncaa_class': 'FR'
        },
        {
            'player_name': 'Mo Bamba', 'overall_pick': 6, 'rsci_rank': 2,
            'height_in': 84.0, 'ape_index': 9.0, 'vertical_in': 30.0, 
            'ncaa_per': 26.4, 'ncaa_bpm': 10.0, 'ncaa_ast_tov_ratio': 0.28, 
            'ncaa_stocks': 8.0, 'ncaa_ts_pct': 0.590, 'ncaa_usg_pct': 20.0, 'ncaa_class': 'FR'
        },
        {
            'player_name': 'Shai Gilgeous-Alexander', 'overall_pick': 11, 'rsci_rank': 31,
            'height_in': 77.5, 'ape_index': 6.0, 'vertical_in': 35.0, 
            'ncaa_per': 20.6, 'ncaa_bpm': 8.5, 'ncaa_ast_tov_ratio': 1.88, 
            'ncaa_stocks': 3.2, 'ncaa_ts_pct': 0.580, 'ncaa_usg_pct': 22.0, 'ncaa_class': 'FR'
        }
    ]
    
    df_2018 = pd.DataFrame(data)
    
    # 🧪 APPLY THE ENGINEERED METRICS
    df_2018['length_playmaking'] = df_2018['ape_index'] * df_2018['ncaa_ast_tov_ratio']
    df_2018['efficiency_per_touch'] = df_2018['ncaa_per'] / (df_2018['ncaa_usg_pct'] + 1)

    X_2018 = pd.get_dummies(df_2018.drop(columns=['player_name']), columns=['ncaa_class'])
    
    for col in model_columns:
        if col not in X_2018.columns:
            X_2018[col] = 0
    X_2018 = X_2018[model_columns]

    print("🔮 Running Predictions...")
    numeric_predictions = model.predict(X_2018)
    df_2018['projection'] = le.inverse_transform(numeric_predictions)
    probs = model.predict_proba(X_2018)
    df_2018['confidence'] = [round(max(p) * 100, 1) for p in probs]

    print("\n--- 🏀 2018 XGBOOST DRAFT BACKTEST ---")
    print(df_2018[['overall_pick', 'player_name', 'projection', 'confidence']].to_string(index=False))

if __name__ == "__main__":
    run_2018_backtest()