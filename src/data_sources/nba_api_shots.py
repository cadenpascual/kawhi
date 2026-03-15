import os
import time
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
import random

# Define custom headers to bypass simple bot detection/firewall
custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

def fetch_league_shots_cached(
    season: str = '2015-16', 
    season_type: str = "Regular Season", 
    context_measure: str = "FGA", 
    max_retries: int = 5
) -> pd.DataFrame:
    """
    Fetches league-wide shot data with local caching and exponential backoff.
    """
    # Define cache path (aligned with your data folder structure)
    cache_dir = "../data/raw/shots"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Standardize filename based on parameters to avoid mixing up seasons/types
    filename_tag = f"{season}_{season_type.replace(' ', '_')}"
    cache_filename = f"{cache_dir}/league_shots_{filename_tag}.csv"
    
    # 1. Check local cache first
    if os.path.exists(cache_filename):
        print(f"[⚡] Loaded {season} {season_type} Shots instantly from local cache.")
        return pd.read_csv(cache_filename)
        
    # 2. Fetch from API if cache doesn't exist
    print(f"[*] Fetching {season} League Shots from NBA API (Sneaking past firewall)...")
    for attempt in range(max_retries):
        try:
            # team_id=0 and player_id=0 fetches league-wide data
            response = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=0,
                season_nullable=season,
                season_type_all_star=season_type,
                context_measure_simple=context_measure,
                headers=custom_headers,
                timeout=300
            )
            df = response.get_data_frames()[0]
            
            # 3. Save result to CSV cache
            df.to_csv(cache_filename, index=False)
            print(f"[✓] Successfully fetched and cached {len(df)} shot records.")
            return df
            
        except Exception as e:
            # Exponential backoff with jitter: (2^attempt) + a random float
            wait_time = (2 ** attempt) + random.random() 
            print(f"[!] API Error: {e}")
            print(f"[!] Retrying in {wait_time:.2f}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)
            
    raise ConnectionError(f"NBA API blocked the request after {max_retries} attempts.")