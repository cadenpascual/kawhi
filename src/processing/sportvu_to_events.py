import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.utils.casting import safe_int, safe_float
from src.tracking.possession import identify_possession
from src.processing.pbp.context import pbp_context
from src.processing.pbp.indexing import build_pbp_index
from src.tracking.event_summaries import event_clock_span, first_ball_xy

# RENAME THIS FUNCTION
def parse_sportvu_kinematics(game_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms unstructured SportVU JSON into a standardized temporal array 
    suitable for dynamical systems modeling.
    """
    if not game_dict or "events" not in game_dict:
        return []

    gameid = safe_int(game_dict.get("gameid"), default=0)
    parsed_events = []

    for ev in game_dict["events"]:
        moments = ev.get("moments", [])
        if not moments: continue

        quarter = safe_int(moments[0][0], default=0) if moments[0] else 0
        
        event_obj = {
            "gameid": gameid,
            "event_id": ev.get("eventId"),
            "event_id_raw": ev.get("eventId"), 
            "quarter": quarter,
            "frames": []
        }

        # Vectorized assembly of the state space per frame
        for frame_id, moment in enumerate(moments, start=1):
            if not moment or len(moment) < 6 or not moment[5]: 
                continue
            
            coords = moment[5]
            
            event_obj["frames"].append({
                "frame_id": frame_id,
                "game_clock": safe_float(moment[2]),
                "shot_clock": safe_float(moment[3]),
                "ball": {
                    "x": safe_float(coords[0][2]),
                    "y": safe_float(coords[0][3]),
                    "z": safe_float(coords[0][4]),
                },
                "players": [
                    {
                        "teamid": safe_int(p[0]),
                        "playerid": safe_int(p[1]),
                        "x": safe_float(p[2]),
                        "y": safe_float(p[3]),
                        "z": safe_float(p[4]),
                    } for p in coords[1:]
                ]
            })

        if event_obj["frames"]:
            parsed_events.append(event_obj)

    return parsed_events


# raw SportVU JSON to processed tracking events
def raw_sportvu_to_tracking_events(game: dict) -> list[dict]:
    """
    Convert raw SportVU 'events' with 'moments' into your standard tracking_events format.
    No PBP join, no possession assignment — just frames with clocks + positions.
    """
    tracking_events = []
    frame_counter = 0
    gameid = int(game["gameid"])

    for ev in game.get("events", []):
        moments = ev.get("moments")
        if not moments:
            continue

        # quarter is in moment[0] (based on your raw format)
        quarter = int(moments[0][0])

        event_obj = {
            "gameid": gameid,
            "event_id_raw": ev.get("eventId"),  # keep raw id if you want
            "quarter": quarter,
            "frames": []
        }

        for moment in moments:
            if moment is None or len(moment) < 6:
                continue

            frame_counter += 1

            ball_row = moment[5][0]   # [-1, -1, x, y, z]
            players_rows = moment[5][1:]  # [teamid, playerid, x, y, z] x10

            frame = {
                "frame_id": frame_counter,
                "game_clock": safe_float(moment[2]),
                "shot_clock": safe_float(moment[3]),
                "ball": {
                    "x": safe_float(moment[5][0][2]),
                    "y": safe_float(moment[5][0][3]),
                    "z": safe_float(moment[5][0][4]),
                },
                "players": [
                    {
                        "teamid": safe_int(p[0]),
                        "playerid": safe_int(p[1]),
                        "x": safe_float(p[2]),
                        "y": safe_float(p[3]),
                        "z": safe_float(p[4]),
                    }
                    for p in moment[5][1:]
                ],
            }

            event_obj["frames"].append(frame)

        if event_obj["frames"]:
            tracking_events.append(event_obj)

    return tracking_events
