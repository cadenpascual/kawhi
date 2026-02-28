import pandas as pd

# Safe casting functions
def safe_int(x, default=None):
    return int(x) if pd.notna(x) else default

def safe_float(x, default=None):
    return float(x) if pd.notna(x) else default
def timestring_to_seconds(s: str):
    if pd.isna(s):
        return None
    m, sec = s.split(":")
    return 60 * int(m) + int(sec)