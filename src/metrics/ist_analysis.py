import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats
from scipy.stats import pearsonr
import time
import os


# ==========================================
# HELPER: ID MAPPERS
# ==========================================
def get_team_map():
    """Returns a dictionary mapping official NBA Team IDs to their full names."""
    return {
        1610612737: 'Atlanta Hawks', 1610612738: 'Boston Celtics',
        1610612739: 'Cleveland Cavaliers', 1610612740: 'New Orleans Pelicans',
        1610612741: 'Chicago Bulls', 1610612742: 'Dallas Mavericks',
        1610612743: 'Denver Nuggets', 1610612744: 'Golden State Warriors',
        1610612745: 'Houston Rockets', 1610612746: 'LA Clippers',
        1610612747: 'Los Angeles Lakers', 1610612748: 'Miami Heat',
        1610612749: 'Milwaukee Bucks', 1610612750: 'Minnesota Timberwolves',
        1610612751: 'Brooklyn Nets', 1610612752: 'New York Knicks',
        1610612753: 'Orlando Magic', 1610612754: 'Indiana Pacers',
        1610612755: 'Philadelphia 76ers', 1610612756: 'Phoenix Suns',
        1610612757: 'Portland Trail Blazers', 1610612758: 'Sacramento Kings',
        1610612759: 'San Antonio Spurs', 1610612760: 'Oklahoma City Thunder',
        1610612761: 'Toronto Raptors', 1610612762: 'Utah Jazz',
        1610612763: 'Memphis Grizzlies', 1610612764: 'Washington Wizards',
        1610612765: 'Detroit Pistons', 1610612766: 'Charlotte Hornets'
    }

def add_team_names(df, id_column='Defensive_Team_ID'):
    """Takes a dataframe and adds a 'Team_Name' column based on the ID."""
    team_map = get_team_map()
    # .map() instantly translates the IDs to names, leaving it blank if not found
    df.insert(1, 'Team_Name', df[id_column].map(team_map)) 
    return df

def get_player_map():
    """Builds a dictionary of {Player_ID: Player_Name} using the offline NBA API static list."""
    nba_players = players.get_players()
    return {player['id']: player['full_name'] for player in nba_players}

def add_player_names(df, id_column='Player_ID'):
    """Takes a dataframe and adds a 'Player_Name' column based on the ID."""
    player_map = get_player_map()
    df.insert(1, 'Player_Name', df[id_column].map(player_map))
    return df

def get_team_abbr(team_name):
    """Helper to turn 'Boston Celtics' into 'BOS' for cleaner charts"""
    if not isinstance(team_name, str): return "UNK"
    # Special cases
    if team_name == 'Los Angeles Lakers': return 'LAL'
    if team_name == 'LA Clippers': return 'LAC'
    if team_name == 'Golden State Warriors': return 'GSW'
    if team_name == 'New York Knicks': return 'NYK'
    if team_name == 'San Antonio Spurs': return 'SAS'
    if team_name == 'New Orleans Pelicans': return 'NOP'
    if team_name == 'Oklahoma City Thunder': return 'OKC'
    return team_name[:3].upper()

# ==========================================
# HELPER: FETCH STATS NBA API
# ==========================================
custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

def fetch_team_stats_cached(season='2015-16', max_retries=5):
    cache_filename = f"../data/raw/defense/nba_team_stats_{season}.csv"
    
    if os.path.exists(cache_filename):
        print(f"[⚡] Loaded {season} Team Stats instantly from local cache.")
        return pd.read_csv(cache_filename)
        
    print(f"[*] Fetching {season} Team Stats from NBA API (Sneaking past firewall)...")
    for attempt in range(max_retries):
        try:
            # We pass the custom headers here!
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season, 
                measure_type_detailed_defense='Advanced',
                headers=custom_headers, 
                timeout=60 
            ).get_data_frames()[0]
            
            team_stats.to_csv(cache_filename, index=False)
            print("[✓] Successfully fetched and cached Team Stats.")
            return team_stats
            
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"[!] Blocked/Timeout. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)
            
    raise Exception("NBA API completely blocked the request.")

def fetch_player_stats_cached(season='2015-16', max_retries=5):
    cache_filename = f"../data/raw/defense/nba_player_stats_{season}.csv"
    
    if os.path.exists(cache_filename):
        print(f"[⚡] Loaded {season} Player Stats instantly from local cache.")
        return pd.read_csv(cache_filename)
        
    print(f"[*] Fetching {season} Player Stats from NBA API (Sneaking past firewall)...")
    for attempt in range(max_retries):
        try:
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, 
                measure_type_detailed_defense='Defense',
                headers=custom_headers,
                timeout=60
            ).get_data_frames()[0]
            
            player_stats.to_csv(cache_filename, index=False)
            print("[✓] Successfully fetched and cached Player Stats.")
            return player_stats
            
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"[!] Blocked/Timeout. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)
            
    raise Exception("NBA API completely blocked the request.")

# ==========================================
# 1. VALIDATE IST PREDICTIVE POWER
# ==========================================

def validate_ist_model(csv_path="../data/processed/shots/all_plays_ist_master.csv", plot=True):
    """
    Groups plays into 5 tiers of Defensive IST pressure to prove 
    that higher IST correlates to a lower Field Goal Percentage.
    """
    df = pd.read_csv(csv_path)
    df['Shot_Made'] = pd.to_numeric(df['Shot_Made'], errors='coerce')
    df_valid = df.dropna(subset=['Shot_Made', 'Real_IST']).copy()

    # Create 5 buckets of defensive pressure
    df_valid['Pressure_Tier'] = pd.qcut(
        df_valid['Real_IST'], q=5, 
        labels=['1. Wide Open', '2. Light Contest', '3. Average', '4. Tight', '5. Smothered']
    )

    # Calculate FG% per tier
    results = df_valid.groupby('Pressure_Tier', observed=True)['Shot_Made'].agg(
        FG_Pct='mean', Sample_Size='count'
    ).reset_index()
    
    results['FG_Pct'] = (results['FG_Pct'] * 100).round(2)

    if plot:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=results, x='Pressure_Tier', y='FG_Pct', palette='coolwarm')
        plt.title("FG% by Real IST Pressure Tier")
        plt.ylabel("Field Goal %")
        plt.xlabel("Real IST Pressure Tier")
        plt.ylim(0, max(results['FG_Pct']) + 10)
        for i, val in enumerate(results['FG_Pct']):
            plt.text(i, val + 1, f"{val}%", ha='center', fontweight='bold')
        plt.show()

    return results

# ==========================================
# 2. TEAM DEFENSIVE RANKINGS
# ==========================================
def analyze_team_sim_ist(ist_csv="../data/processed/shots/all_plays_ist_master.csv", stats_csv="../data/raw/defense/nba_team_stats_2015-16.csv", min_possessions=500):
    """
    Reads the master IST dataset, ranks NBA teams by deviation,
    loads the manually downloaded NBA stats CSV, and plots the correlation.
    """
    # ==========================================
    # 1. PROCESS YOUR RAW IST DATA
    # ==========================================
    df = pd.read_csv(ist_csv)
    df['Shot_Made'] = pd.to_numeric(df['Shot_Made'], errors='coerce')
    
    team_defense = df.groupby('Defensive_Team_ID').agg(
        Possessions=('Play_Index', 'count'),
        Avg_Real_IST=('Real_IST', 'mean'),
        Avg_Sim_IST=('Sim_IST', 'mean'),
        Total_Deviation=('Pressure_Prevented', 'sum'), 
        Opponent_FG_Pct=('Shot_Made', 'mean')
    ).reset_index()

    team_defense = team_defense[team_defense['Possessions'] >= min_possessions].copy()
    team_defense['Deviation_Per_100'] = ((team_defense['Total_Deviation'] / team_defense['Possessions']) * 100).round(2)
    team_defense['Opponent_FG_Pct'] = (team_defense['Opponent_FG_Pct'] * 100).round(2)
    team_defense['Avg_Real_IST'] = team_defense['Avg_Real_IST'].round(2)

    # Add Team Names mapping (Assumes your get_team_map() function is loaded)
    team_map = get_team_map()
    team_defense.insert(1, 'Team_Name', team_defense['Defensive_Team_ID'].map(team_map))
    team_defense = team_defense.sort_values(by='Deviation_Per_100', ascending=True)

    # ==========================================
    # 2. LOAD & MERGE YOUR MANUALLY DOWNLOADED CSV
    # ==========================================
    if not os.path.exists(stats_csv):
        print(f"Error: Could not find {stats_csv}. Make sure it is in your folder!")
        return None

    print(f"Loading {stats_csv}...")
    team_stats = pd.read_csv(stats_csv)
    
    # Grab just the Team Name and their Defensive Rating
    team_stats = team_stats[['TEAM', 'DefRtg']]
    
    # Merge using their NAMES!
    merged_df = pd.merge(team_defense, team_stats, left_on='Team_Name', right_on='TEAM', how='inner')

    # ==========================================
    # 3. PLOT THE DATA
    # ==========================================
    plt.figure(figsize=(12, 8))
    
    # Plot the dots using the exact CSV column name 'DefRtg'
    sns.scatterplot(data=merged_df, x='Deviation_Per_100', y='DefRtg', s=120, color='crimson')
    
    # Annotate the dots with team abbreviations (Assumes get_team_abbr() is loaded)
    for i, row in merged_df.iterrows():
        abbr = get_team_abbr(row['Team_Name']) 
        plt.text(row['Deviation_Per_100'], row['DefRtg'] + 0.2, abbr, fontsize=9, ha='center')

    # Draw the regression line
    sns.regplot(data=merged_df, x='Deviation_Per_100', y='DefRtg', scatter=False, color='black', line_kws={'linestyle':'dashed', 'alpha':0.4})

    # Formatting
    plt.title("Team Defense: Model Deviation vs 2015-16 Defensive Rating", fontsize=15, fontweight='bold')
    plt.xlabel("Deviation from Optimal JKO Model Per 100 Plays (Higher = Suboptimal/Worse)", fontsize=12)
    plt.ylabel("Official NBA Defensive Rating (Lower = Better Defense)", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.show()
        
    return merged_df

def analyze_team_real_ist(ist_csv="../data/processed/shots/all_plays_ist_master.csv", stats_csv="../data/raw/defense/nba_team_stats_2015-16.csv", min_possessions=500):
    """
    Reads the CSV, calculates the Average Real IST per team, loads the manually 
    downloaded NBA Defensive Ratings, and plots them to validate the base physics equation.
    """
    # ==========================================
    # 1. PROCESS THE RAW CSV
    # ==========================================
    df = pd.read_csv(ist_csv)
    df['Shot_Made'] = pd.to_numeric(df['Shot_Made'], errors='coerce')
    
    # Aggregate to get Avg_Real_IST
    team_defense = df.groupby('Defensive_Team_ID').agg(
        Possessions=('Play_Index', 'count'),
        Avg_Real_IST=('Real_IST', 'mean'),
    ).reset_index()

    team_defense = team_defense[team_defense['Possessions'] >= min_possessions].copy()
    team_defense['Avg_Real_IST'] = team_defense['Avg_Real_IST'].round(2)

    # Add Team Names (Assumes your get_team_map() function is loaded)
    team_map = get_team_map()
    team_defense.insert(1, 'Team_Name', team_defense['Defensive_Team_ID'].map(team_map))

    # ==========================================
    # 2. LOAD & MERGE YOUR MANUALLY DOWNLOADED CSV
    # ==========================================
    if not os.path.exists(stats_csv):
        print(f"Error: Could not find {stats_csv}. Make sure it is in your folder!")
        return None

    print(f"Loading {stats_csv} for Real IST validation...")
    team_stats = pd.read_csv(stats_csv)
    
    # Grab just the Team Name and their Defensive Rating
    team_stats = team_stats[['TEAM', 'DefRtg']]
    
    # Merge using their NAMES!
    merged_df = pd.merge(team_defense, team_stats, left_on='Team_Name', right_on='TEAM', how='inner')

    # ==========================================
    # 3. CALCULATE CORRELATION STATS
    # ==========================================
    # Calculate Pearson correlation (r) and p-value
    r_value, p_value = pearsonr(merged_df['Avg_Real_IST'], merged_df['DefRtg'])
    r_squared = r_value ** 2

    # ==========================================
    # 4. PLOT THE CHART
    # ==========================================
    plt.figure(figsize=(12, 8))
    
    # Plotting Avg_Real_IST on the X-axis and DefRtg on the Y-axis
    sns.scatterplot(data=merged_df, x='Avg_Real_IST', y='DefRtg', s=120, color='darkorange')
    
    # Annotate the dots with team abbreviations (Assumes get_team_abbr() is loaded)
    for i, row in merged_df.iterrows():
        abbr = get_team_abbr(row['Team_Name'])
        plt.text(row['Avg_Real_IST'], row['DefRtg'] + 0.2, abbr, fontsize=9, ha='center')

    # Add trendline
    sns.regplot(data=merged_df, x='Avg_Real_IST', y='DefRtg', scatter=False, color='black', line_kws={'linestyle':'dashed', 'alpha':0.4})

    plt.title("Mean Real IST vs. NBA Defensive Rating", fontsize=15, fontweight='bold')
    # Corrected label here:
    plt.xlabel("Average Real IST per Play (Lower = More Topological Pressure)", fontsize=12)
    plt.ylabel("Official NBA Defensive Rating (Lower = Better Defense)", fontsize=12)
    
    # ==========================================
    # 5. ADD THE STATISTICS BOX
    # ==========================================
    stats_text = f"Correlation (r): {r_value:.3f}\n$R^2$: {r_squared:.3f}\np-value: {p_value:.1e}"
    
    # Placed in the top-left corner (x=0.05, y=0.95)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    plt.grid(True, alpha=0.3)
    plt.show()
    
    return merged_df
# ==========================================
# 3. INDIVIDUAL PLAYER IMPACT (ON-COURT)
# ==========================================
def analyze_player_sim_ist(ist_csv="../data/processed/shots/all_plays_ist_master.csv", stats_csv="../data/raw/defense/nba_player_stats_2015-16.csv", min_plays=500):
    """
    Reads the master IST dataset, ranks NBA players by deviation,
    loads the manually downloaded NBA stats CSV, and plots the correlation.
    """
    # ==========================================
    # 1. PROCESS YOUR RAW IST DATA
    # ==========================================
    df = pd.read_csv(ist_csv)
    
    defender_cols = ['Defender_1_PID', 'Defender_2_PID', 'Defender_3_PID', 'Defender_4_PID', 'Defender_5_PID']
    melted_df = df.melt(
        id_vars=['Game_File', 'Play_Index', 'Real_IST', 'Pressure_Prevented', 'Defensive_Team_ID'], 
        value_vars=defender_cols, 
        value_name='Player_ID'
    )
    
    melted_df = melted_df.dropna(subset=['Player_ID']).copy()
    melted_df['Player_ID'] = melted_df['Player_ID'].astype(int)

    player_impact = melted_df.groupby('Player_ID').agg(
        Plays_Contested=('Play_Index', 'count'),
        Avg_Deviation=('Pressure_Prevented', 'mean'), 
        Primary_Team_ID=('Defensive_Team_ID', lambda x: x.mode()[0] if not x.empty else None)
    ).reset_index()

    player_impact = player_impact[player_impact['Plays_Contested'] >= min_plays].copy()

    # Add Names mapping (Make sure your get_player_map() function from earlier is loaded!)
    player_map = get_player_map() 
    team_map = get_team_map()     
    
    player_impact.insert(1, 'Player_Name', player_impact['Player_ID'].map(player_map))
    player_impact.insert(2, 'Team_Name', player_impact['Primary_Team_ID'].map(team_map))

    player_impact = player_impact.sort_values(by='Avg_Deviation', ascending=True)
    
    # ==========================================
    # 2. LOAD & MERGE YOUR MANUALLY DOWNLOADED CSV
    # ==========================================
    if not os.path.exists(stats_csv):
        print(f"Error: Could not find {stats_csv}. Make sure it is in your folder!")
        return None

    print(f"Loading {stats_csv}...")
    player_stats = pd.read_csv(stats_csv)
    
    # Clean the column headers (removes weird newlines like "DEF\nWS")
    player_stats.columns = player_stats.columns.str.replace('\n', ' ').str.strip()
    
    # Keep only the Name and the Defensive Win Shares column
    player_stats = player_stats[['Player', 'DEF WS']]
    
    # Merge using their NAMES instead of IDs!
    merged_df = pd.merge(player_impact, player_stats, left_on='Player_Name', right_on='Player', how='inner')

    # ==========================================
    # 3. PLOT THE DATA
    # ==========================================
    plt.figure(figsize=(13, 8))
    sns.scatterplot(
        data=merged_df, x='Avg_Deviation', y='DEF WS', 
        size='Plays_Contested', sizes=(20, 250), alpha=0.6, color='royalblue'
    )
    
    # Annotate Top 15 Most Optimal Defenders
    top_optimal = merged_df.sort_values(by='Avg_Deviation', ascending=True).head(15)
    for i, row in top_optimal.iterrows():
        last_name = str(row['Player_Name']).split()[-1] if len(str(row['Player_Name']).split()) > 1 else row['Player_Name']
        team_abbr = get_team_abbr(row['Team_Name']) # Assumes get_team_abbr() is loaded
        label = f"{last_name} ({team_abbr})"
        plt.text(row['Avg_Deviation'] + 0.005, row['DEF WS'], label, fontsize=9, ha='left', va='center', fontweight='bold')

    sns.regplot(data=merged_df, x='Avg_Deviation', y='DEF WS', scatter=False, color='black', line_kws={'linestyle':'dashed', 'alpha':0.4})

    plt.title(f"Individual Defense: Model Deviation vs 2015-16 Defensive Win Shares", fontsize=15, fontweight='bold')
    plt.xlabel("Average Unit Deviation from Optimal Model (Higher = Suboptimal/Worse)", fontsize=12)
    plt.ylabel("Official Defensive Win Shares (Higher = Better Defense)", fontsize=12)
    
    plt.legend(title="Sample Size (Plays)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return merged_df

def analyze_player_win_shares(ist_csv="../data/processed/shots/all_plays_ist_master.csv", stats_csv="../data/raw/defense/nba_player_stats_2015-16.csv", min_plays=500):
    """
    Reads the master IST dataset, ranks NBA players by Average Real IST,
    loads the manually downloaded NBA stats CSV, and plots the correlation
    to validate the base physics model at the player level.
    """
    # ==========================================
    # 1. PROCESS YOUR RAW IST DATA
    # ==========================================
    df = pd.read_csv(ist_csv)
    
    defender_cols = ['Defender_1_PID', 'Defender_2_PID', 'Defender_3_PID', 'Defender_4_PID', 'Defender_5_PID']
    melted_df = df.melt(
        id_vars=['Game_File', 'Play_Index', 'Real_IST', 'Defensive_Team_ID'], 
        value_vars=defender_cols, 
        value_name='Player_ID'
    )
    
    melted_df = melted_df.dropna(subset=['Player_ID']).copy()
    melted_df['Player_ID'] = melted_df['Player_ID'].astype(int)

    # Group by Player and calculate their Average REAL IST
    player_impact = melted_df.groupby('Player_ID').agg(
        Plays_Contested=('Play_Index', 'count'),
        Avg_Real_IST=('Real_IST', 'mean'), 
        Primary_Team_ID=('Defensive_Team_ID', lambda x: x.mode()[0] if not x.empty else None)
    ).reset_index()

    player_impact = player_impact[player_impact['Plays_Contested'] >= min_plays].copy()

    # Add Names mapping 
    player_map = get_player_map() 
    team_map = get_team_map()     
    
    player_impact.insert(1, 'Player_Name', player_impact['Player_ID'].map(player_map))
    player_impact.insert(2, 'Team_Name', player_impact['Primary_Team_ID'].map(team_map))

    # SORT BY LOWEST: Lower Real IST = More Topological Pressure
    player_impact = player_impact.sort_values(by='Avg_Real_IST', ascending=True)
    
    # ==========================================
    # 2. LOAD & MERGE YOUR MANUALLY DOWNLOADED CSV
    # ==========================================
    if not os.path.exists(stats_csv):
        print(f"Error: Could not find {stats_csv}. Make sure it is in your folder!")
        return None

    print(f"Loading {stats_csv} for Real IST validation...")
    player_stats = pd.read_csv(stats_csv)
    
    # Clean the column headers (removes weird newlines like "DEF\nWS")
    player_stats.columns = player_stats.columns.str.replace('\n', ' ').str.strip()
    
    # Keep only the Name and the Defensive Win Shares column
    player_stats = player_stats[['Player', 'DEF WS']]
    
    # Merge using their NAMES
    merged_df = pd.merge(player_impact, player_stats, left_on='Player_Name', right_on='Player', how='inner')

    # ==========================================
    # 3. PLOT THE DATA
    # ==========================================
    plt.figure(figsize=(13, 8))
    
    # Plot Avg_Real_IST on the X-axis
    sns.scatterplot(
        data=merged_df, x='Avg_Real_IST', y='DEF WS', 
        size='Plays_Contested', sizes=(20, 250), alpha=0.6, color='darkorange'
    )
    
    # Annotate Top 15 Players with the MOST pressure (Lowest Avg_Real_IST)
    top_pressure = merged_df.sort_values(by='Avg_Real_IST', ascending=True).head(15)
    for i, row in top_pressure.iterrows():
        last_name = str(row['Player_Name']).split()[-1] if len(str(row['Player_Name']).split()) > 1 else row['Player_Name']
        team_abbr = get_team_abbr(row['Team_Name']) 
        label = f"{last_name} ({team_abbr})"
        
        plt.text(row['Avg_Real_IST'] + 0.05, row['DEF WS'], label, fontsize=9, ha='left', va='center', fontweight='bold')

    sns.regplot(data=merged_df, x='Avg_Real_IST', y='DEF WS', scatter=False, color='black', line_kws={'linestyle':'dashed', 'alpha':0.4})

    plt.title("Base Model Validation: Avg Real IST vs 2015-16 Defensive Win Shares", fontsize=15, fontweight='bold')
    plt.xlabel("Average Real IST per Play (Lower = More Topological Pressure)", fontsize=12)
    plt.ylabel("Official Defensive Win Shares (Higher = Better Defense)", fontsize=12)
    
    plt.legend(title="Sample Size (Plays)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return merged_df

def analyze_player_def_rating(ist_csv="../data/processed/shots/all_plays_ist_master.csv", stats_csv="../data/raw/defense/nba_player_stats_2015-16.csv", min_plays=500):
    """
    Reads the master IST dataset, ranks NBA players by Average Real IST,
    loads the manually downloaded NBA stats CSV, and plots the correlation
    against individual Defensive Rating, including statistical strength.
    """
    # ==========================================
    # 1. PROCESS YOUR RAW IST DATA
    # ==========================================
    df = pd.read_csv(ist_csv)
    
    defender_cols = ['Defender_1_PID', 'Defender_2_PID', 'Defender_3_PID', 'Defender_4_PID', 'Defender_5_PID']
    melted_df = df.melt(
        id_vars=['Game_File', 'Play_Index', 'Real_IST', 'Defensive_Team_ID'], 
        value_vars=defender_cols, 
        value_name='Player_ID'
    )
    
    melted_df = melted_df.dropna(subset=['Player_ID']).copy()
    melted_df['Player_ID'] = melted_df['Player_ID'].astype(int)

    player_impact = melted_df.groupby('Player_ID').agg(
        Plays_Contested=('Play_Index', 'count'),
        Avg_Real_IST=('Real_IST', 'mean'), 
        Primary_Team_ID=('Defensive_Team_ID', lambda x: x.mode()[0] if not x.empty else None)
    ).reset_index()

    player_impact = player_impact[player_impact['Plays_Contested'] >= min_plays].copy()

    # Add Names mapping (Make sure your get_player_map & get_team_map functions are run first!)
    player_map = get_player_map() 
    team_map = get_team_map()     
    
    player_impact.insert(1, 'Player_Name', player_impact['Player_ID'].map(player_map))
    player_impact.insert(2, 'Team_Name', player_impact['Primary_Team_ID'].map(team_map))

    player_impact = player_impact.sort_values(by='Avg_Real_IST', ascending=True)
    
    # ==========================================
    # 2. LOAD & MERGE YOUR MANUALLY DOWNLOADED CSV
    # ==========================================
    if not os.path.exists(stats_csv):
        print(f"Error: Could not find {stats_csv}. Make sure it is in your folder!")
        return None

    print(f"Loading {stats_csv} for Real IST vs Def Rtg validation...")
    player_stats = pd.read_csv(stats_csv)
    
    player_stats.columns = player_stats.columns.str.replace('\n', ' ').str.strip()
    player_stats = player_stats[['Player', 'DEF RTG']]
    merged_df = pd.merge(player_impact, player_stats, left_on='Player_Name', right_on='Player', how='inner')

    # ==========================================
    # 3. CALCULATE CORRELATION STATS
    # ==========================================
    # Calculate Pearson correlation (r) and p-value
    r_value, p_value = pearsonr(merged_df['Avg_Real_IST'], merged_df['DEF RTG'])
    r_squared = r_value ** 2

    # ==========================================
    # 4. PLOT THE DATA
    # ==========================================
    plt.figure(figsize=(13, 8))
    
    sns.scatterplot(
        data=merged_df, x='Avg_Real_IST', y='DEF RTG', 
        size='Plays_Contested', sizes=(20, 250), alpha=0.6, color='purple'
    )
    
    top_pressure = merged_df.sort_values(by='Avg_Real_IST', ascending=True).head(15)
    for i, row in top_pressure.iterrows():
        last_name = str(row['Player_Name']).split()[-1] if len(str(row['Player_Name']).split()) > 1 else row['Player_Name']
        team_abbr = get_team_abbr(row['Team_Name']) 
        label = f"{last_name} ({team_abbr})"
        plt.text(row['Avg_Real_IST'] + 0.05, row['DEF RTG'], label, fontsize=9, ha='left', va='center', fontweight='bold')

    sns.regplot(data=merged_df, x='Avg_Real_IST', y='DEF RTG', scatter=False, color='black', line_kws={'linestyle':'dashed', 'alpha':0.4})

    plt.title("Base Model Validation: Avg Real IST vs 2015-16 Individual Def Rating", fontsize=15, fontweight='bold')
    plt.xlabel("Average Real IST per Play (Lower = More Topological Pressure)", fontsize=12)
    plt.ylabel("Official Individual Defensive Rating (Lower = Better Defense)", fontsize=12)
    
    # ==========================================
    # 5. ADD THE STATISTICS BOX TO THE CHART
    # ==========================================
    # Create the text string for the box
    stats_text = f"Correlation (r): {r_value:.3f}\n$R^2$: {r_squared:.3f}\np-value: {p_value:.1e}"
    
    # Place the box in the bottom right corner (0.95, 0.05) so it doesn't block the top-left elite defenders
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.legend(title="Sample Size (Plays)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return merged_df

# ==========================================
# 4. INDIVIDUAL GAME IMPACT
# ==========================================

def analyze_game_real_ist(ist_csv="../data/processed/shots/all_plays_ist_master.csv", boxscore_csv="../data/raw/defense/nba_game_scores_2015-16.csv"):
    """
    Analyzes game-by-game performance using IST normalized to 100 possessions.
    """
    df = pd.read_csv(ist_csv)
    
    # Extract Date and Map Teams
    df['Game Date'] = df['Game_File'].astype(str).str.extract(r'(\d{2}\.\d{2}\.\d{4})')[0]
    team_map = get_team_map()
    df['Team'] = df['Defensive_Team_ID'].map(team_map).apply(get_team_abbr)
    
    # 1. Calculate Total IST and Play Count per Game
    game_ist = df.groupby(['Team', 'Game Date']).agg(
        Plays_Tracked=('Play_Index', 'count'),
        Total_Real_IST=('Real_IST', 'sum') 
    ).reset_index()
    
    # 2. NORMALIZE: (Total IST / Plays) * 100
    # This gives us the IST 'Cost' per 100 tracked plays
    game_ist['IST_per_100'] = (game_ist['Total_Real_IST'] / game_ist['Plays_Tracked']) * 100
    
    # Filter for significance
    game_ist = game_ist[game_ist['Plays_Tracked'] >= 20].copy()

    # Load and clean Boxscores
    boxscores = pd.read_csv(boxscore_csv)
    boxscores.columns = [col.strip() for col in boxscores.columns]
    
    # Aggressive Cleaning for Merge
    game_ist['Clean_Date'] = pd.to_datetime(game_ist['Game Date'])
    boxscores['Clean_Date'] = pd.to_datetime(boxscores['Game Date'])
    game_ist['Clean_Team'] = game_ist['Team'].str.strip().str.upper()
    boxscores['Clean_Team'] = boxscores['Team'].str.strip().str.upper()

    merged = pd.merge(game_ist, boxscores, on=['Clean_Team', 'Clean_Date'], how='inner')

    # 3. Correlation against PT_DIFF
    r_value, p_value = pearsonr(merged['IST_per_100'], merged['PT_DIFF'])
    
    # PLOT
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=merged, x='IST_per_100', y='PT_DIFF', hue='W/L', 
                    palette={'W': 'forestgreen', 'L': 'crimson'}, alpha=0.6)
    
    sns.regplot(data=merged, x='IST_per_100', y='PT_DIFF', scatter=False, color='black', line_kws={'linestyle':'--'})

    plt.title("Pace-Normalized Validation: IST per 100 Plays vs Point Differential", fontsize=14, fontweight='bold')
    plt.xlabel("Real IST per 100 Defensive Plays (Lower = More Pressure)", fontsize=12)
    plt.ylabel("Final Point Differential (PT_DIFF)", fontsize=12)
    
    # Stats Box
    stats_text = f"Correlation (r): {r_value:.3f}\n$R^2$: {r_value**2:.3f}\np-value: {p_value:.1e}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.show()
    return merged

def analyze_game_simulation_deviation(ist_csv="../data/processed/shots/all_plays_ist_master.csv", boxscore_csv="../data/raw/defense/nba_game_scores_2015-16.csv"):
    """
    Analyzes game-by-game performance by calculating how much a team
    deviated from the JKO Optimal Simulation, normalized per 100 plays.
    """
    df = pd.read_csv(ist_csv)
    
    # 1. Extract Date and Map Teams
    df['Game Date'] = df['Game_File'].astype(str).str.extract(r'(\d{2}\.\d{2}\.\d{4})')[0]
    team_map = get_team_map()
    df['Team'] = df['Defensive_Team_ID'].map(team_map).apply(get_team_abbr)
    
    # 2. Calculate Pressure Prevented (Real - Sim) and Plays
    # Note: 'Pressure_Prevented' is often already (Real_IST - Sim_IST)
    game_agg = df.groupby(['Team', 'Game Date']).agg(
        Plays_Tracked=('Play_Index', 'count'),
        Total_Deviation=('Pressure_Prevented', 'sum') 
    ).reset_index()
    
    # 3. NORMALIZE: Deviation per 100 Plays
    game_agg['Dev_per_100'] = (game_agg['Total_Deviation'] / game_agg['Plays_Tracked']) * 100
    
    # Filter for significance
    game_agg = game_agg[game_agg['Plays_Tracked'] >= 20].copy()

    # 4. Load and Clean Boxscores
    boxscores = pd.read_csv(boxscore_csv)
    boxscores.columns = [col.strip() for col in boxscores.columns]
    
    # Aggressive Date/Team Cleaning for the Merge
    game_agg['Clean_Date'] = pd.to_datetime(game_agg['Game Date'])
    boxscores['Clean_Date'] = pd.to_datetime(boxscores['Game Date'])
    game_agg['Clean_Team'] = game_agg['Team'].str.strip().str.upper()
    boxscores['Clean_Team'] = boxscores['Team'].str.strip().str.upper()

    merged = pd.merge(game_agg, boxscores, on=['Clean_Team', 'Clean_Date'], how='inner')

    # 5. Calculate Correlation against PT_DIFF
    r_value, p_value = pearsonr(merged['Dev_per_100'], merged['PT_DIFF'])
    
    # 6. PLOT
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=merged, x='Dev_per_100', y='PT_DIFF', hue='W/L', 
                    palette={'W': 'forestgreen', 'L': 'crimson'}, alpha=0.6, s=100)
    
    sns.regplot(data=merged, x='Dev_per_100', y='PT_DIFF', scatter=False, color='black', line_kws={'linestyle':'--'})

    plt.title("Simulation Validation: Normalized Deviation vs Point Differential", fontsize=14, fontweight='bold')
    plt.xlabel("Inefficiency Gap (Real IST - Sim IST) per 100 Plays\n(Lower = Closer to Optimal Positioning)", fontsize=12)
    plt.ylabel("Final Point Differential (PT_DIFF)", fontsize=12)
    
    # Add Statistics Box
    stats_text = f"Correlation (r): {r_value:.3f}\n$R^2$: {r_value**2:.3f}\np-value: {p_value:.1e}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.grid(True, alpha=0.2)
    plt.show()
    
    return merged

def analyze_game_ist_vs_pts_allowed(ist_csv="../data/processed/shots/all_plays_ist_master.csv", boxscore_csv="../data/raw/defense/nba_game_scores_2015-16.csv"):
    """
    Plots Game-Level Real IST vs Points Allowed.
    Uses PTS and PT_DIFF to derive the opponent's scoring.
    """
    # 1. Load IST Data
    df = pd.read_csv(ist_csv)
    
    # Extract Date from Game_File (e.g., traj_01.23.2016.ATL...)
    df['Game Date'] = df['Game_File'].astype(str).str.extract(r'(\d{2}\.\d{2}\.\d{4})')[0]
    
    # Map Teams (Assumes get_team_map and get_team_abbr are available)
    team_map = get_team_map()
    df['Team'] = df['Defensive_Team_ID'].map(team_map).apply(get_team_abbr)
    
    # Aggregate IST per Game
    game_ist = df.groupby(['Team', 'Game Date']).agg(
        Plays_Tracked=('Play_Index', 'count'),
        Avg_Real_IST=('Real_IST', 'mean')
    ).reset_index()
    
    # 2. Load Boxscore Data
    boxscores = pd.read_csv(boxscore_csv)
    boxscores.columns = [col.strip() for col in boxscores.columns]
    
    # DERIVE DEFENSIVE OUTCOME: Opponent PTS = PTS - PT_DIFF
    # Example: Scored 95, Diff -3 -> Opponent scored 98.
    boxscores['Opponent_PTS'] = boxscores['PTS'] - boxscores['PT_DIFF']
    
    # Normalize by Minutes for Overtime accuracy
    boxscores['Pts_Allowed_Per_Min'] = boxscores['Opponent_PTS'] / boxscores['MIN']

    # 3. Clean and Merge
    game_ist['Clean_Date'] = pd.to_datetime(game_ist['Game Date'])
    boxscores['Clean_Date'] = pd.to_datetime(boxscores['Game Date'])
    game_ist['Clean_Team'] = game_ist['Team'].str.strip().str.upper()
    boxscores['Clean_Team'] = boxscores['Team'].str.strip().str.upper()

    merged = pd.merge(game_ist, boxscores, on=['Clean_Team', 'Clean_Date'], how='inner')

    # 4. Correlation Analysis
    r_value, p_value = pearsonr(merged['Avg_Real_IST'], merged['Pts_Allowed_Per_Min'])

    # 5. Plotting
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=merged, x='Avg_Real_IST', y='Pts_Allowed_Per_Min', 
                    alpha=0.5, color='#e74c3c', s=60)
    
    sns.regplot(data=merged, x='Avg_Real_IST', y='Pts_Allowed_Per_Min', 
                scatter=False, color='black', line_kws={'linestyle':'--', 'alpha':0.6})

    plt.title("Defensive Validation: Topological 'Openness' vs Points Allowed", fontsize=15, fontweight='bold')
    plt.xlabel("Average Real IST (Higher = Offense is 'More Open')", fontsize=12)
    plt.ylabel("Points Allowed Per Minute (Lower = Better Defense)", fontsize=12)

    # Statistics Box
    stats_text = (f"Correlation (r): {r_value:.3f}\n"
                  f"$R^2$: {r_value**2:.3f}\n"
                  f"p-value: {p_value:.1e}")
    
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.grid(True, alpha=0.2)
    plt.show()
    
    return merged