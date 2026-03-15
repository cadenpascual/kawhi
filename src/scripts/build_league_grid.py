from src.data_sources.nba_api_shots import fetch_league_shots
from src.features.shots.shot_maps import contested_fg_grids
from src.data_io.grids import save_grids

def main():
    season = "2015-16"
    shots = fetch_league_shots(season)
    grids = contested_fg_grids(shots, n_bins=10)
    save_grids(grids, season=season, out_dir="data/processed/grids")

if __name__ == "__main__":
    main()