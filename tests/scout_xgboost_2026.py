import pandas as pd
import joblib
import numpy as np

def run_final_2026_board():
    print("🧠 Loading the Final AI Brain (Age + Pedigree Optimized)...")
    model = joblib.load('models/nba_xgboost_model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    model_columns = joblib.load('models/xgboost_columns.pkl')

    print("📊 Loading 2026 Tankathon Board...")
    df_2026 = pd.read_csv('data/tankathon_2026_full_board.csv')
    
    # --- 1. BASELINE INJECTIONS FOR THE WHOLE BOARD ---
    # Give everyone league average physicals
    df_2026['height_in'] = 78.0 
    df_2026['ape_index'] = 3.5  
    df_2026['vertical_in'] = 32.0
    
    # Assign Draft Age based on College Class
    age_map = {'FR': 19.5, 'SO': 20.5, 'JR': 21.5, 'SR': 22.5}
    df_2026['draft_age'] = df_2026['ncaa_class'].map(age_map).fillna(20.0)
    
    # Assign a baseline RSCI rank (Proxy: assume their draft rank roughly mirrors their high school rank)
    df_2026['rsci_rank'] = df_2026['overall_pick'] * 1.5

    # --- 2. SURGICAL INJECTIONS FOR THE TOP 5 PHENOMS ---
    # AJ Dybantsa (RSCI: 1, Age: 19.4)
    df_2026.loc[df_2026['player_name'] == 'AJ Dybantsa', 
                ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 3.0, 38.0, 1, 19.4]
    
    # Darryn Peterson (RSCI: 3, Age: 19.6)
    df_2026.loc[df_2026['player_name'] == 'Darryn Peterson', 
                ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [77.0, 5.0, 36.0, 3, 19.6]
    
    # Cam Boozer (RSCI: 2, Age: 18.9 - Extremely young!)
    df_2026.loc[df_2026['player_name'] == 'Cam Boozer', 
                ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 4.0, 35.0, 2, 18.9]
    
    # Caleb Wilson (RSCI: 4, Age: 19.8)
    df_2026.loc[df_2026['player_name'] == 'Caleb Wilson', 
                ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 3.0, 37.0, 4, 19.8]
    
    # Darius Acuff Jr. (RSCI: 5, Age: 19.5)
    df_2026.loc[df_2026['player_name'] == 'Darius Acuff Jr.', 
                ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [75.0, 4.0, 34.0, 5, 19.5]

    # --- 3. PRE-PROCESSING ---
    X_2026 = pd.get_dummies(df_2026.drop(columns=['player_name']), columns=['ncaa_class'])
    
    # Align columns perfectly with what XGBoost expects
    for col in model_columns:
        if col not in X_2026.columns:
            X_2026[col] = 0
    X_2026 = X_2026[model_columns]

    # --- 4. PREDICTIONS ---
    print("🔮 Running the Final Analytics Engine...")
    numeric_predictions = model.predict(X_2026)
    df_2026['projection'] = le.inverse_transform(numeric_predictions)
    
    probs = model.predict_proba(X_2026)
    df_2026['confidence'] = [round(max(p) * 100, 1) for p in probs]

    # --- 5. THE REVEAL ---
    print("\n=======================================================")
    print("     🏀 OFFICIAL 2026 AI DRAFT BOARD (TOP 10) 🏀       ")
    print("=======================================================")
    top_10 = df_2026.head(10)
    print(top_10[['overall_pick', 'player_name', 'projection', 'confidence']].to_string(index=False))

    print("\n🌟 THE FRANCHISE SAVIORS (PROJECTED SUPERSTARS):")
    superstars = df_2026[df_2026['projection'] == 'Superstar']
    if not superstars.empty:
        print(superstars[['overall_pick', 'player_name', 'confidence']].to_string(index=False))
    else:
        print("None. The AI does not believe any player meets the Superstar threshold.")

    print("\n💎 THE DATA DARLINGS (FRONT OFFICE STEALS):")
    steals = df_2026[df_2026['projection'] == 'Steal']
    if not steals.empty:
        # Sort by confidence so the most guaranteed steals are at the top
        steals = steals.sort_values(by='confidence', ascending=False).head(8)
        print(steals[['overall_pick', 'player_name', 'confidence']].to_string(index=False))
    else:
        print("None.")
    print("=======================================================\n")

if __name__ == "__main__":
    run_final_2026_board()