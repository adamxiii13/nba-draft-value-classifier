# 🏀 NBA Draft AI Scout: Machine Learning Value Classifier

An end-to-end Machine Learning pipeline and interactive Front Office Dashboard designed to project the NBA success of college prospects. 

This project goes beyond traditional box-score scouting by feeding an **Extreme Gradient Boosting (XGBoost)** algorithm 25 years of historical data—including college efficiency, physical combine measurements, draft age, and high school recruiting pedigree (RSCI).

## 🧠 Methodology & Target Labels
The model is designed to evaluate a prospect's pure mathematical profile and project them into one of five historical value categories based on their Draft Pick and Career Value Over Expectation (VOE):

* **🌟 Superstar:** Multi-time All-NBA / MVP caliber.
* **⭐ Star:** Perennial starter / All-Star appearances.
* **💎 Steal:** Outperformed their draft slot by >700 career minutes per season.
* **📊 Expected:** Returned average, replacement-level value for their draft slot.
* **🚩 Bust:** Top-14 Lottery pick who fell severely short of historical minute/production expectations.

## ⚙️ Key Features
* **XGBoost Classifier:** Tuned with a max-depth of 6 and a learning rate of 0.05 to prevent overfitting.
* **SMOTE Balancing:** Uses Synthetic Minority Over-sampling to train the AI on extremely rare outlier profiles (Superstars) without biasing the model towards busts.
* **Feature Engineering:** Accounts for the "Unranked Penalty" (non-Top 100 high school recruits) and imputes exact Draft Age to punish older prospects with inflated counting stats.
* **Streamlit Dashboard:** A dynamic UI allowing scouts to tweak physicals, college stats, and draft slots to run real-time "What-If" scenarios.
* **Draft Slot Pressure Relief:** A custom tool to remove the mathematical burden of being a Top-3 pick, allowing the AI to evaluate pure basketball talent in a vacuum.

## 📂 Repository Structure
```text
nba-draft-value-classifier/
│
├── data/                       # Contains raw CSVs (Ignored by .gitignore)
├── models/                     # Saved XGBoost .pkl files (Ignored by .gitignore)
├── sql/
│   └── 01_build_database_schema.sql  # SQL logic defining Busts/Steals and joining raw data
│
├── src/
│   ├── scrape_pedigree.py      # BeautifulSoup scraper for historical RSCI Top 100 ranks
│   ├── scrape_ages.py          # BeautifulSoup scraper for exact NBA Draft Age
│   └── train_xgboost.py        # Core ML pipeline (Data merge, SMOTE, Training, Export)
│
├── app.py                      # Interactive Streamlit Front Office Dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation