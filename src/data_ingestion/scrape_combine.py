import pandas as pd
import time
from nba_api.stats.endpoints import draftcombinestats
import os

def convert_to_inches(measurement):
    """Handles both "6' 5.25" strings and raw decimal formats"""
    if pd.isna(measurement) or str(measurement).strip() == '':
        return None
    
    # If it's already a number (like 77.25), just return it
    try:
        val = float(measurement)
        return val
    except ValueError:
        pass # It's a string like "6' 5.25''", so continue to parse

    # Parse the string format
    try:
        parts = str(measurement).split("'")
        feet = float(parts[0])
        inches = float(parts[1].replace('"', '').replace("''", "").strip())
        return (feet * 12) + inches
    except:
        return None

def fetch_combine_data():
    all_seasons_data = []
    
    for year in range(2001, 2026): 
        season = f"{year}-{str(year+1)[-2:]}"
        print(f"📡 Fetching {season} Combine Data...")
        try:
            combine = draftcombinestats.DraftCombineStats(season_all_time=season)
            df = combine.get_data_frames()[0]
            df['SEASON'] = year
            all_seasons_data.append(df)
            time.sleep(1.5) 
        except Exception as e:
            print(f"⚠️ Could not fetch data for {season}: {e}")

    print("\n⚙️ Aggregating and Cleaning Data...")
    full_combine = pd.concat(all_seasons_data, ignore_index=True)

    full_combine['HEIGHT_WO_SHOES'] = full_combine['HEIGHT_WO_SHOES'].apply(convert_to_inches)
    full_combine['WINGSPAN'] = full_combine['WINGSPAN'].apply(convert_to_inches)
    
    rename_map = {
        'PLAYER_NAME': 'player_name',
        'SEASON': 'season',
        'HEIGHT_WO_SHOES': 'height_no_shoes',
        'WEIGHT': 'weight',
        'WINGSPAN': 'wingspan',
        'STANDING_VERTICAL_LEAP': 'standing_vertical',
        'MAX_VERTICAL_LEAP': 'max_vertical',
        'LANE_AGILITY_TIME': 'lane_agility',
        'MODIFIED_14_CRAZY_SPRINT': 'three_quarter_sprint'
    }
    
    if 'THREE_QUARTER_SPRINT' in full_combine.columns and 'MODIFIED_14_CRAZY_SPRINT' not in full_combine.columns:
        rename_map['THREE_QUARTER_SPRINT'] = 'three_quarter_sprint'

    available_cols = [col for col in rename_map.keys() if col in full_combine.columns]
    final_df = full_combine[available_cols].rename(columns=rename_map)

    if not os.path.exists('data'):
        os.makedirs('data')
        
    output_path = 'data/historical_combine.csv'
    final_df.to_csv(output_path, index=False)
    print(f"✅ Successfully extracted {len(final_df)} player measurements to {output_path}")

if __name__ == "__main__":
    fetch_combine_data()