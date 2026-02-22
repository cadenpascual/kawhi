import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.cwd().parent
DATA_DIR = ROOT / "data"

def load_tracking_json(game_id):
    path = DATA_DIR / "tracking" / f"{game_id}.json"
    with open(path, "r") as f:
        return json.load(f)

def save_dataframe(df, name):
    path = DATA_DIR / "processed" / f"{name}.parquet"
    df.to_parquet(path)

def load_dataframe(name):
    path = DATA_DIR / "processed" / f"{name}.parquet"
    return pd.read_parquet(path)

def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def save_json(path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=indent)