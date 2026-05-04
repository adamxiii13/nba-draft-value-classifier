import pandas as pd
import time
from nba_api.stats.endpoints import playercareerstats
import os

def fetch_career_stats(draft_df):
    print("Fetching career stats for drafted players. This will take a few minutes...")
    all_stats = []
    
    # Get the unique player IDs from our draft history CSV
    player_ids = draft_df['person_id'].dropna().unique()
    
    for i, player_id in enumerate(player_ids):
        try:
            # Removed custom headers to avoid the firewall, kept the 60-second timeout
            career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=60)
            df = career.get_data_frames()[0] 
            
            df['person_id'] = player_id
            all_stats.append(df)
            
            if (i + 1) % 50 == 0:
                print(f"Fetched {i + 1} / {len(player_ids)} players...")
                
            # The sleep timer is our real protection against getting blocked
            time.sleep(0.6)
            
        except Exception as e:
            print(f"Error fetching player {player_id}: {e}")
            time.sleep(2) 
            
    final_df = pd.concat(all_stats, ignore_index=True)
    final_df.columns = final_df.columns.str.lower()
    return final_df

if __name__ == "__main__":
    input_path = "data/raw/nba_draft_history.csv"
    
    # Check if the file actually exists before trying to read it
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. Run data_fetch.py first.")
    else:
        draft_df = pd.read_csv(input_path)
        
        career_df = fetch_career_stats(draft_df)
        
        output_path = "data/raw/nba_career_stats.csv"
        career_df.to_csv(output_path, index=False)
        
        print(f"Successfully saved career stats to {output_path}")