"""
NBA Draft Value Classifier - Front Office Dashboard
===================================================
This Streamlit application serves as the interactive UI for the XGBoost draft model.
It allows users to select prospects from the 2026 draft class, upload custom CSVs, 
and tweak physical/statistical metrics in real-time to see how the AI's projections shift.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Draft Scout", layout="wide", page_icon="🏀")

# --- DIRECTORIES ---
DATA_DIR = 'data'
MODEL_DIR = 'models'

# --- LOAD AI ---
@st.cache_resource
def load_ai():
    model_path = os.path.join(MODEL_DIR, 'nba_xgboost_model.pkl')
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    cols_path = os.path.join(MODEL_DIR, 'xgboost_columns.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(le_path) and os.path.exists(cols_path)):
        st.error("⚠️ Model files not found! Please run `python src/train_xgboost.py` to generate the AI brain before launching the dashboard.")
        st.stop()
        
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    cols = joblib.load(cols_path)
    return model, le, cols

model, le, model_columns = load_ai()

# --- LOAD DEFAULT DATA ---
@st.cache_data
def load_board():
    board_path = os.path.join(DATA_DIR, 'tankathon_2026_full_board.csv')
    if os.path.exists(board_path):
        df = pd.read_csv(board_path)
        
        # Inject baselines so the sliders have a default state
        if 'height_in' not in df.columns: df['height_in'] = 78.0
        if 'ape_index' not in df.columns: df['ape_index'] = 3.5
        if 'vertical_in' not in df.columns: df['vertical_in'] = 32.0
        if 'rsci_rank' not in df.columns: df['rsci_rank'] = df['overall_pick'] * 1.5
        
        # Assign Draft Age based on NCAA Class
        if 'draft_age' not in df.columns: 
            age_map = {'FR': 19.5, 'SO': 20.5, 'JR': 21.5, 'SR': 22.5}
            df['draft_age'] = df['ncaa_class'].map(age_map).fillna(20.0)
            
        # Manually inject the top 5 known prospects (Can be updated post-combine)
        df.loc[df['player_name'] == 'AJ Dybantsa', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 3.0, 38.0, 1, 19.4]
        df.loc[df['player_name'] == 'Darryn Peterson', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [77.0, 5.0, 36.0, 3, 19.6]
        df.loc[df['player_name'] == 'Cam Boozer', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 4.0, 35.0, 2, 18.9]
        df.loc[df['player_name'] == 'Caleb Wilson', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [81.0, 3.0, 37.0, 4, 19.8]
        df.loc[df['player_name'] == 'Darius Acuff Jr.', ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age']] = [75.0, 4.0, 34.0, 5, 19.5]
        
        return df
    return pd.DataFrame()

# --- MAIN UI ---
st.title("🏀 AI Draft Scout: Front Office Dashboard")
st.markdown("Select a prospect from the board or upload your own scouting CSV. Tweak the metrics below to test 'What-If' scenarios.")

# --- SIDEBAR: PLAYER SELECTION ---
st.sidebar.header("1. Board Selection")
uploaded_file = st.sidebar.file_uploader("Upload Custom CSV (Optional)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Ensure no crashes if custom CSV is missing specialized columns
    for col in ['height_in', 'ape_index', 'vertical_in', 'rsci_rank', 'draft_age', 'overall_pick']:
        if col not in df.columns:
            df[col] = 0.0
else:
    df = load_board()

if df.empty:
    st.error(f"Could not load player data. Ensure `tankathon_2026_full_board.csv` exists in the `{DATA_DIR}` folder.")
    st.stop()

player_list = df['player_name'].tolist()
selected_player = st.sidebar.selectbox("Select Prospect", player_list)

# Get the specific data for the chosen player to act as slider defaults
p_data = df[df['player_name'] == selected_player].iloc[0]

# --- SIDEBAR: TWEAK METRICS ---
st.sidebar.header("2. Tweak Metrics")

# Safe helper to handle missing data from custom CSVs gracefully
def safe_float(val, default=0.0):
    try: return float(val)
    except: return float(default)

overall_pick = st.sidebar.slider("Drafted At (Pick #)", 1, 60, int(safe_float(p_data.get('overall_pick', 15))))

# 🔥 THE PRESSURE RELIEF VALVE 🔥
evaluate_in_vacuum = st.sidebar.checkbox(
    "Remove Draft Slot Pressure", 
    value=False, 
    help="Forces the AI to evaluate the player's pure talent as Pick #30, removing the extreme historical penalty of being a top lottery pick."
)

# Trick the AI into evaluating them as the last pick of the 1st round if checked
ai_evaluated_pick = overall_pick
if evaluate_in_vacuum:
    ai_evaluated_pick = 30

st.sidebar.subheader("Context")
classes = ["FR", "SO", "JR", "SR"]
current_class = p_data.get('ncaa_class', 'FR')
class_idx = classes.index(current_class) if current_class in classes else 0
ncaa_class = st.sidebar.selectbox("Class", classes, index=class_idx)
draft_age = st.sidebar.slider("Draft Age", 18.0, 25.0, safe_float(p_data.get('draft_age', 20.0)))
rsci_rank = st.sidebar.slider("HS RSCI Rank", 1, 300, int(safe_float(p_data.get('rsci_rank', 150))))

st.sidebar.subheader("Physicals")
height_in = st.sidebar.slider("Height (Inches)", 70.0, 90.0, safe_float(p_data.get('height_in', 78.0)))
ape_index = st.sidebar.slider("Ape Index", -5.0, 15.0, safe_float(p_data.get('ape_index', 3.5)))
vertical_in = st.sidebar.slider("Max Vertical", 20.0, 50.0, safe_float(p_data.get('vertical_in', 32.0)))

st.sidebar.subheader("College Stats")
ncaa_per = st.sidebar.slider("PER", 0.0, 45.0, safe_float(p_data.get('ncaa_per', 15.0)))
ncaa_bpm = st.sidebar.slider("BPM", -5.0, 20.0, safe_float(p_data.get('ncaa_bpm', 5.0)))
ncaa_ast_tov_ratio = st.sidebar.slider("AST/TOV Ratio", 0.0, 5.0, safe_float(p_data.get('ncaa_ast_tov_ratio', 1.0)))
ncaa_stocks = st.sidebar.slider("Stocks (STL% + BLK%)", 0.0, 20.0, safe_float(p_data.get('ncaa_stocks', 3.0)))
ncaa_ts_pct = st.sidebar.slider("True Shooting %", 0.300, 0.800, safe_float(p_data.get('ncaa_ts_pct', 0.550)))
ncaa_usg_pct = st.sidebar.slider("Usage %", 10.0, 45.0, safe_float(p_data.get('ncaa_usg_pct', 20.0)))

# --- PREDICTION LOGIC ---
input_dict = {
    'overall_pick': ai_evaluated_pick, 'height_in': height_in, 'ape_index': ape_index,
    'vertical_in': vertical_in, 'ncaa_per': ncaa_per, 'ncaa_bpm': ncaa_bpm,
    'ncaa_ast_tov_ratio': ncaa_ast_tov_ratio, 'ncaa_stocks': ncaa_stocks,
    'ncaa_ts_pct': ncaa_ts_pct, 'ncaa_usg_pct': ncaa_usg_pct,
    'ncaa_class': ncaa_class, 'rsci_rank': rsci_rank, 'draft_age': draft_age
}

df_input = pd.DataFrame([input_dict])
X_input = pd.get_dummies(df_input, columns=['ncaa_class'])

# Ensure all columns exist and are ordered correctly
for col in model_columns:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[model_columns]

# Force strict float typing to prevent XGBoost silent boolean failures
X_input = X_input.astype(float)

numeric_pred = model.predict(X_input)
grade = le.inverse_transform(numeric_pred)[0]
probs = model.predict_proba(X_input)[0]
confidence = round(max(probs) * 100, 1)

# --- DISPLAY ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"Prospect: {selected_player}")
    if grade == 'Superstar':
        st.success(f"### AI Grade: 🌟 SUPERSTAR ({confidence}% Confidence)")
    elif grade == 'Steal':
        st.info(f"### AI Grade: 💎 DRAFT STEAL ({confidence}% Confidence)")
    elif grade == 'Star':
        st.success(f"### AI Grade: ⭐ STAR ({confidence}% Confidence)")
    elif grade == 'Expected':
        st.warning(f"### AI Grade: 📊 EXPECTED VALUE ({confidence}% Confidence)")
    else:
        st.error(f"### AI Grade: 🚩 BUST ({confidence}% Confidence)")
        
    if evaluate_in_vacuum:
        st.markdown(f"**Analytics Note:** Draft Pressure is **REMOVED**. The AI is grading this prospect assuming they are drafted at **Pick #30**.")
    else:
        st.markdown(f"**Analytics Note:** The AI is grading this prospect assuming they are drafted at **Pick #{overall_pick}**.")

with col2:
    st.write("### Adjusted Profile")
    # Show the actual pick the user selected in the JSON output, not the hidden AI pick
    display_dict = input_dict.copy()
    display_dict['overall_pick'] = overall_pick
    st.json(display_dict)