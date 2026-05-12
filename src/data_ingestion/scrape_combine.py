"""
NBA Draft Value Classifier - Combine Scraper
============================================
Fetches historical NBA Combine physicals via the nba_api. 
Automatically merges new measurements (Height, Ape Index, Max Vertical) 
into the live Streamlit 2026 Draft Board.
"""

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
    
    # THE FIX: Increased range to 2027 so Python successfully loops through 2026
    for year in range(2001, 2027): 
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

    # --- NEW SECTION: AUTO-UPDATE THE STREAMLIT 2026 DRAFT BOARD ---
    board_path = 'data/tankathon_2026_full_board.csv'
    if os.path.exists(board_path):
        print("\n🔄 Merging new 2026 combine data into the Streamlit Draft Board...")
        board_df = pd.read_csv(board_path)
        
        # Extract just the 2026 class from the combine scrape
        combine_2026 = final_df[final_df['season'] == 2026].copy()
        
        # Calculate Ape Index safely
        combine_2026['new_ape_index'] = combine_2026['wingspan'] - combine_2026['height_no_shoes']
        
        # Keep only the columns we need to update
        combine_updates = combine_2026[['player_name', 'height_no_shoes', 'new_ape_index', 'max_vertical']].copy()
        combine_updates.rename(columns={
            'height_no_shoes': 'new_height',
            'max_vertical': 'new_vertical'
        }, inplace=True)
        
        # Merge with the draft board
        merged = pd.merge(board_df, combine_updates, on='player_name', how='left')
        
        # Initialize default columns if they don't exist
        if 'height_in' not in merged.columns: merged['height_in'] = 78.0
        if 'ape_index' not in merged.columns: merged['ape_index'] = 3.5
        if 'vertical_in' not in merged.columns: merged['vertical_in'] = 32.0
        
        # Overwrite estimations with official Combine physicals (fallback to old data if player skipped combine)
        merged['height_in'] = merged['new_height'].fillna(merged['height_in'])
        merged['ape_index'] = merged['new_ape_index'].fillna(merged['ape_index'])
        merged['vertical_in'] = merged['new_vertical'].fillna(merged['vertical_in'])
        
        # Clean up temporary columns
        merged.drop(columns=['new_height', 'new_ape_index', 'new_vertical'], inplace=True)
        
        merged.to_csv(board_path, index=False)
        print(f"💎 Successfully updated {board_path} with official physicals!")

if __name__ == "__main__":
    fetch_combine_data()