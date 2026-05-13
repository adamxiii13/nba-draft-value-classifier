import pandas as pd
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

print("🧠 Loading AI and Explainer Module...")
model = joblib.load('models/nba_xgboost_model.pkl')
features = joblib.load('models/xgboost_columns.pkl')

try:
    le = joblib.load('models/label_encoder.pkl')
    has_encoder = True
except:
    has_encoder = False

# --- LOAD THE DATA ---
df = pd.read_csv('data/tankathon_2026_full_board.csv') 

print("📊 Applying Surgical Data Injections...")

# --- 1. BASELINE INJECTIONS ---
df['height_in'] = 78.0 
df['ape_index'] = 3.5  
df['vertical_in'] = 32.0
age_map = {'FR': 19.5, 'SO': 20.5, 'JR': 21.5, 'SR': 22.5}
df['draft_age'] = df['ncaa_class'].map(age_map).fillna(20.0)
df['rsci_rank'] = df['overall_pick'] * 1.5

# --- 2. SURGICAL INJECTIONS FOR THE TOP 5 ---
df.loc[df['player_name'] == 'AJ Dybantsa', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 3.0, 38.0, 1, 19.4]
df.loc[df['player_name'] == 'Darryn Peterson', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [77.0, 5.0, 36.0, 3, 19.6]
df.loc[df['player_name'] == 'Cam Boozer', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 4.0, 35.0, 2, 18.9]
df.loc[df['player_name'] == 'Caleb Wilson', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 3.0, 37.0, 4, 19.8]
df.loc[df['player_name'] == 'Darius Acuff Jr.', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [75.0, 4.0, 34.0, 5, 19.5]

# --- 3. PRE-PROCESSING ---
X_2026 = pd.get_dummies(df.drop(columns=['player_name']), columns=['ncaa_class'])
for col in features:
    if col not in X_2026.columns:
        X_2026[col] = 0
X_2026 = X_2026[features]

print("🔍 Running SHAP Explainer... (This might take a few seconds)")

with open("AI_Exact_Data_Report.txt", "w", encoding="utf-8") as f:
    
    # Loop through the players using the fully injected data
    for index, row in df.iterrows():
        player_name = row['player_name']
        X_player = X_2026.iloc[[index]]
        
        # Get prediction
        pred_probs = model.predict_proba(X_player)[0]
        max_prob_index = pred_probs.argmax()
        
        if has_encoder:
            predicted_class = le.inverse_transform([max_prob_index])[0]
        else:
            predicted_class = model.classes_[max_prob_index]
            
        confidence = pred_probs[max_prob_index] * 100
        
        f.write(f"=========================================\n")
        f.write(f" 🏀 AI SCOUTING REPORT: {player_name} \n")
        f.write(f"=========================================\n")
        f.write(f"Overall Projection: {predicted_class} ({confidence:.1f}% Confidence)\n\n")
        f.write("🔍 ALL DRIVING FACTORS:\n")
        
        # SHAP Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_player)
        
        if isinstance(shap_values, list):
            class_shap_values = shap_values[max_prob_index][0]
        else:
            class_shap_values = shap_values[0, :, max_prob_index]
            
        feature_impacts = list(zip(features, class_shap_values))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, impact in feature_impacts:
            direction = "PUSHED UP" if impact > 0 else "PULLED DOWN"
            if impact != 0:
                f.write(f"-> {feature}: {direction} the {predicted_class} probability. (Impact factor: {abs(impact):.3f})\n")
        
        f.write("\n\n")

print("✅ Done! Open 'AI_Exact_Data_Report.txt' to read the true report.")