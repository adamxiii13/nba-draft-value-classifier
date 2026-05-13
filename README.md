# 🏀 NBA Draft AI Scout: Machine Learning Value Classifier (2026 Edition)

An end-to-end Machine Learning pipeline and interactive Front Office Dashboard designed to project the NBA success of college prospects. 

This project goes beyond traditional box-score scouting by feeding an **Extreme Gradient Boosting (XGBoost)** algorithm 25 years of historical data—including college efficiency, physical combine measurements, draft age, and high school recruiting pedigree (RSCI).

## 🧠 Methodology & Target Labels
The model evaluates a prospect's pure mathematical profile and projects them into one of five historical value categories based on their Draft Pick and Career Value Over Expectation (VOE):

* 🌟 **Superstar:** Multi-time All-NBA / MVP caliber.
* ⭐ **Star:** Perennial starter / All-Star appearances.
* 💎 **Steal:** Outperformed their draft slot by >700 career minutes per season.
* 📊 **Expected:** Returned average, replacement-level value for their draft slot.
* 🚩 **Bust:** Top-14 Lottery pick who fell severely short of historical minute/production expectations.

## ⚙️ Key Features
* **XGBoost Classifier:** Tuned with a max-depth of 6 and a learning rate of 0.05 to prevent overfitting.
* **SMOTE Balancing:** Uses Synthetic Minority Over-sampling to train the AI on extremely rare outlier profiles (Superstars) without biasing the model towards busts.
* **Feature Engineering:** Accounts for the "Unranked Penalty" (non-Top 100 high school recruits) and imputes exact Draft Age to punish older prospects with inflated counting stats.
* **Streamlit Dashboard:** A dynamic UI allowing scouts to tweak physicals, college stats, and draft slots to run real-time "What-If" scenarios.
* **Draft Slot Pressure Relief:** A custom tool to remove the mathematical burden of being a Top-3 pick, allowing the AI to evaluate pure basketball talent in a vacuum.
* **SHAP AI Explainer:** (explain_player.py) A diagnostic script that cracks open the XGBoost black box to show exactly *why* a player received their grade (e.g., showing if their assist-to-turnover ratio "Pushed Up" or "Pulled Down" their projection).

---

## 🚀 Quick Start (Running the Dashboard)

Because the pre-trained models and master datasets are too large for standard GitHub commits, they are ignored via `.gitignore`. **To run this out of the box without building a database, you need to download the release files.**

**1. Clone the repository:**
```bash
git clone [https://github.com/adamxiii13/nba-draft-value-classifier.git](https://github.com/adamxiii13/nba-draft-value-classifier.git)
cd nba-draft-value-classifier
