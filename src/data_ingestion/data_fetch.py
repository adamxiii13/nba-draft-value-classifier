import pandas as pd
import time
from nba_api.stats.endpoints import drafthistory
import os

def fetch_draft_history(start_year=2010, end_year=2021):
    """
    Fetches NBA draft history from the nba_api.
    We cap it at 2021 to ensure players have had a few years in the league 
    to evaluate their 'value'.
    """
    print(f"Fetching draft history from {start_year} to {end_year}...")
    
    # Rely on the package's built-in headers, but keep the 60-second patience
    draft_data = drafthistory.DraftHistory(timeout=60)
    
    # Get the data as a pandas DataFrame
    df = draft_data.get_data_frames()[0]
    
    # Filter the DataFrame to our target years
    df['SEASON'] = df['SEASON'].astype(int)
    filtered_df = df[(df['SEASON'] >= start_year) & (df['SEASON'] <= end_year)]
    
    # Clean up column names (lowercase is generally standard for SQL ingestion)
    filtered_df.columns = filtered_df.columns.str.lower()
    
    return filtered_df

if __name__ == "__main__":
    # Ensure the data directories exist before saving
    os.makedirs("data/raw", exist_ok=True)
    
    # Execute the function
    draft_df = fetch_draft_history()
    
    # Save to our raw data folder
    output_path = "data/raw/nba_draft_history.csv"
    draft_df.to_csv(output_path, index=False)
    
    print(f"Successfully saved {len(draft_df)} draft records to {output_path}")