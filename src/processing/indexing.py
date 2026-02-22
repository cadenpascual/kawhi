# src/processing/indexing.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

def build_tracking_time_index(tracking_events: list[dict]) -> pd.DataFrame:
    rows = []
    for k, ev in enumerate(tracking_events):
        frames = ev.get("frames", [])
        if not frames:
            continue

        gc_raw = [fr.get("game_clock", np.nan) for fr in frames]
        gc = np.asarray(pd.to_numeric(gc_raw, errors="coerce"), dtype=float)
        valid = ~np.isnan(gc)
        if valid.sum() < 2:
            continue

        gc_v = gc[valid]

        gc_start = float(np.max(gc_v))
        gc_end   = float(np.min(gc_v))
        gc_center = 0.5 * (gc_start + gc_end)
        gc_span = gc_start - gc_end

        # monotonicity check (countdown should mostly decrease)
        d = np.diff(gc_v)
        mono_frac = float(np.mean(d <= 0)) if len(d) else np.nan

        rows.append({
            "gameid": int(ev.get("gameid")) if ev.get("gameid") is not None else None,
            "quarter": int(ev.get("quarter")) if ev.get("quarter") is not None else None,
            "event_list_idx": k,
            "gc_start": gc_start,
            "gc_end": gc_end,
            "gc_center": gc_center,
            "gc_span": float(gc_span),
            "n_frames_total": int(len(frames)),
            "n_frames_gc": int(valid.sum()),
            "gc_monotone_frac": mono_frac,
        })

    df = pd.DataFrame(rows)
    # Optional: drop clearly broken segments
    if not df.empty:
        df = df[(df["gc_span"] > 0) & (df["n_frames_gc"] >= 10)]
    return df.reset_index(drop=True)


# VECTORIZED MATCHING 

def attach_tracking_events_interval(shots_g, event_index, span_pad=5.0):
    """
    Vectorized matching of shots to their corresponding tracking event 
    based on game clock intervals. 
    
    Includes fix for 'KeyError: shot_index' by using temporary IDs.
    """
    shots = shots_g.copy()
    events = event_index.copy()

    # 1. Type Safety
    shots["GAME_ID"] = shots["GAME_ID"].astype(int)
    shots["PERIOD"] = shots["PERIOD"].astype(int)
    events["gameid"] = events["gameid"].astype(int)
    events["quarter"] = events["quarter"].astype(int)
    events["gc_start"] = pd.to_numeric(events["gc_start"], errors="coerce")
    events["gc_end"] = pd.to_numeric(events["gc_end"], errors="coerce")
    events = events.dropna(subset=["gc_start", "gc_end"])

    merged_results = []
    
    # 2. Group by Game & Period to minimize Cross-Join size
    groups = shots.groupby(["GAME_ID", "PERIOD"])
    
    for (gid, qtr), shots_sub in groups:
        events_sub = events[
            (events["gameid"] == gid) & 
            (events["quarter"] == qtr)
        ]
        
        if events_sub.empty:
            shots_sub = shots_sub.copy()
            shots_sub["event_list_idx"] = np.nan
            merged_results.append(shots_sub)
            continue
            
        # 3. Cross Merge Setup
        shots_sub = shots_sub.copy()
        shots_sub["_join_key"] = 1
        # Create a temporary unique ID for this chunk to safe-guard deduplication
        shots_sub["_tmp_id"] = np.arange(len(shots_sub))
        
        events_sub = events_sub.copy()
        events_sub["_join_key"] = 1
        
        # Merge
        cross = pd.merge(
            shots_sub,
            events_sub[["event_list_idx", "gc_start", "gc_end", "_join_key"]],
            on="_join_key"
        )
        
        # 4. Logic Constraints
        # Event starts at least 3s before shot (buildup)
        cond_buildup = cross["gc_start"] >= (cross["game_clock"] + 3.0)
        # Event ends near the shot (proximity)
        cond_proximity = cross["gc_end"] <= (cross["game_clock"] + span_pad)
        
        valid = cross[cond_buildup & cond_proximity].copy()
        
        if valid.empty:
            shots_sub.drop(columns=["_join_key", "_tmp_id"], inplace=True)
            shots_sub["event_list_idx"] = np.nan
            merged_results.append(shots_sub)
            continue

        # 5. Deduplicate (Keep closest match)
        valid["time_diff"] = (valid["gc_end"] - valid["game_clock"]).abs()
        
        best_matches = (
            valid.sort_values("time_diff")
            .drop_duplicates(subset=["_tmp_id"]) # Uses our temp ID, not "shot_index"
        )
        
        # 6. Map back results using the temporary ID
        mapping = best_matches.set_index("_tmp_id")["event_list_idx"]
        shots_sub["event_list_idx"] = shots_sub["_tmp_id"].map(mapping)
        
        # Cleanup
        shots_sub.drop(columns=["_join_key", "_tmp_id"], inplace=True)
        merged_results.append(shots_sub)

    if not merged_results:
        return shots_g
        
    return pd.concat(merged_results)