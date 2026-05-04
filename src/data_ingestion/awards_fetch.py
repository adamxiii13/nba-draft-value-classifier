import pandas as pd
import time
from nba_api.stats.endpoints import playerawards
import os

def fetch_player_awards(draft_df):
    print("Fetching awards for drafted players. This will take a few minutes...")
    all_awards = []
    
    player_ids = draft_df['person_id'].dropna().unique()
    
    for i, player_id in enumerate(player_ids):
        try:
            # We hit the playerawards endpoint instead of career stats
            awards = playerawards.PlayerAwards(player_id=player_id, timeout=60)
            df = awards.get_data_frames()[0] 
            
            if not df.empty:
                df['person_id'] = player_id
                all_awards.append(df)
            
            if (i + 1) % 50 == 0:
                print(f"Fetched {i + 1} / {len(player_ids)} players...")
                
            time.sleep(0.6)
            
        except Exception as e:
            # If a player has never won an award, it will just quietly skip them
            pass
            
    if all_awards:
        final_df = pd.concat(all_awards, ignore_index=True)
        final_df.columns = final_df.columns.str.lower()
        return final_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    input_path = "data/raw/nba_draft_history.csv"
    draft_df = pd.read_csv(input_path)
    
    awards_df = fetch_player_awards(draft_df)
    
    if not awards_df.empty:
        output_path = "data/raw/nba_player_awards.csv"
        awards_df.to_csv(output_path, index=False)
        print(f"Successfully saved player awards to {output_path}")
    else:
        print("No awards found.")