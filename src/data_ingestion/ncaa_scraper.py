import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import time
import os
import re
import io # Required for modern Pandas versions

def generate_player_slug(name):
    """Standardizes names and removes suffixes (Jr, III, etc.)"""
    name = name.lower()
    name = re.sub(r"\b(jr|sr|iii|iv|ii)\b", "", name)
    name = re.sub(r"[.'-]", "", name)
    name = " ".join(name.split())
    return name.replace(" ", "-")

def scrape_ncaa_stats(player_name):
    slug = generate_player_slug(player_name)
    url = f"https://www.sports-reference.com/cbb/players/{slug}-1.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            return "404"
        if response.status_code == 429:
            return "429"
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look in comments first (where Sports-Ref hides the good stuff)
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        table = None
        
        for comment in comments:
            if 'id="players_advanced"' in str(comment):
                comment_soup = BeautifulSoup(str(comment), 'html.parser')
                table = comment_soup.find('table', {'id': 'players_advanced'})
                break
        
        # Fallback to main page if not in comments
        if not table:
            table = soup.find('table', {'id': 'players_advanced'})

        if table:
            # PRO FIX: Wrap the table string in io.StringIO
            # This forces Pandas to treat it as HTML content
            table_html = str(table)
            df = pd.read_html(io.StringIO(table_html))[0]
            
            career_row = df[df['Season'] == 'Career'].copy()
            if not career_row.empty:
                career_row['player_name'] = player_name
                return career_row

    except Exception as e:
        # We'll print a shorter error now that we know the logic works
        print(f"  Error: {str(e)[:50]}...")
        
    return None

if __name__ == "__main__":
    input_path = "data/raw/nba_draft_history.csv"
    draft_df = pd.read_csv(input_path)
    players = draft_df['player_name'].unique()
    
    # Let's run the test batch again
    test_batch = players
    results = []

    print(f"Starting Pro NCAA Scrape for {len(test_batch)} players...")

    for i, player in enumerate(test_batch):
        print(f"[{i+1}/{len(test_batch)}] Scraping {player}...", end=" ", flush=True)
        stats = scrape_ncaa_stats(player)
        
        if isinstance(stats, pd.DataFrame):
            results.append(stats)
            print("✓ Found!")
        elif stats == "404":
            print("x (404)")
        elif stats == "429":
            print("!! (429 - Waiting 60s)")
            time.sleep(60)
        else:
            print("x (No table)")
            
        time.sleep(3.1) # Respect the 20-req-per-minute rule

    if results:
        final_df = pd.concat(results, ignore_index=True)
        # Sanitize columns for SQL
        final_df.columns = [c.lower().replace('%', '_pct').replace('/', '_') for c in final_df.columns]
        os.makedirs("data/raw", exist_ok=True)
        final_df.to_csv("data/raw/ncaa_scraped_stats.csv", index=False)
        print(f"\nSuccess! Saved {len(final_df)} records to data/raw/ncaa_scraped_stats.csv")