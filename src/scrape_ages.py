"""
NBA Draft Value Classifier - Draft Age Scraper
==============================================
Scrapes the Basketball-Reference Rookie pages to extract the exact 
age of every player during their rookie season. It then subtracts 
1.0 year to approximate their age on Draft Night.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

DATA_DIR = 'data'

def get_draft_ages(year):
    print(f"📡 Fetching Rookie Ages for {year}...")
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_rookies.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='rookies')
        
        if not table:
            return pd.DataFrame()
            
        players = []
        tbody = table.find('tbody')
        
        for row in tbody.find_all('tr'):
            # Use 'string=' instead of 'text=' to avoid bs4 deprecation warnings
            if 'thead' in row.get('class', []) or row.find('th', string='Rk'): 
                continue
            
            player_td = row.find('td', {'data-stat': 'player'})
            age_td = row.find('td', {'data-stat': 'age'})
            
            if player_td and age_td and age_td.text.strip():
                try:
                    # Rookie age is age on Feb 1st of rookie year.
                    # Subtract 1 to estimate draft night age.
                    draft_age = float(age_td.text.strip()) - 1.0
                    players.append({
                        'player_name': player_td.text.strip(),
                        'draft_age': draft_age
                    })
                except ValueError:
                    continue
                    
        return pd.DataFrame(players)
    
    except Exception as e:
        print(f"⚠️ Error fetching {year}: {e}")
        return pd.DataFrame()

def run_age_scraper():
    all_ages = []
    # Fetch ages from 2000 to 2024
    for year in range(2000, 2025): 
        df = get_draft_ages(year)
        if not df.empty:
            all_ages.append(df)
        time.sleep(3.5) # Prevent IP bans

    if not all_ages:
        print("❌ Scraper failed to pull any data.")
        return

    final_df = pd.concat(all_ages, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['player_name'])
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    output_path = os.path.join(DATA_DIR, 'historical_draft_ages.csv')
    final_df.to_csv(output_path, index=False)
    print(f"✅ SUCCESS! Saved {len(final_df)} ages to {output_path}")

if __name__ == "__main__":
    run_age_scraper()