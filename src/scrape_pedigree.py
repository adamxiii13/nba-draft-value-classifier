"""
NBA Draft Value Classifier - High School Pedigree Scraper
=========================================================
Scrapes Sports-Reference to extract the final RSCI 
(Recruiting Services Consensus Index) Top 100 ranking for 
prospects to gauge their pre-college pedigree.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

DATA_DIR = 'data'

def get_rsci_data(year):
    print(f"📡 Fetching RSCI Top 100 for {year}...")
    url = f"https://www.sports-reference.com/cbb/awards/men/rsci-recruit-rankings-{year}.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"⚠️ Error: Got status code {response.status_code} for {year}")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Locate the specific RSCI table
        table = soup.find('table', id='rsci-rankings')
        
        if not table:
            print(f"⚠️ No table found for {year}.")
            return pd.DataFrame()
            
        players = []
        tbody = table.find('tbody')
        if not tbody: 
            return pd.DataFrame()
        
        # Loop through every row in the table
        for row in tbody.find_all('tr'):
            if 'thead' in row.get('class', []): 
                continue
            
            # Extract Rank (Removing the 'T' for ties, e.g., 'T12' -> '12')
            rank_th = row.find('th', {'data-stat': 'rank'})
            if not rank_th: 
                continue
            rank_text = rank_th.text.strip().replace('T', '')
            
            # Extract Player Name
            player_td = row.find('td', {'data-stat': 'player'})
            if not player_td: 
                continue
            player_name = player_td.text.strip()
            
            if player_name and rank_text and rank_text.isdigit():
                players.append({
                    'player_name': player_name,
                    'rsci_rank': int(rank_text),
                    'rsci_year': year
                })
                
        return pd.DataFrame(players)
    
    except Exception as e:
        print(f"⚠️ Error fetching {year}: {e}")
        return pd.DataFrame()

def run_scraper():
    all_ranks = []
    # Start from 2005 onwards since earlier data is sparse
    for year in range(2005, 2026):
        df = get_rsci_data(year)
        if not df.empty:
            all_ranks.append(df)
            
        time.sleep(3.5) # Prevent IP bans

    if not all_ranks:
        print("❌ Scraper failed to pull any data.")
        return

    final_df = pd.concat(all_ranks, ignore_index=True)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    output_path = os.path.join(DATA_DIR, 'historical_rsci.csv')
    final_df.to_csv(output_path, index=False)
    print(f"✅ SUCCESS! Saved {len(final_df)} rankings to {output_path}")

if __name__ == "__main__":
    run_scraper()