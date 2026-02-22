import numpy as np

def find_release_frame_idx(
    event_frames,
    shot_game_clock,
    *,
    fps=25,
    match="closest",          # "closest" or "prev"
    max_time_diff=1.0,        # seconds; return None if farther than this
    require_shot_clock=False  # if True, drop frames missing shot_clock
):
    """
    Find the release frame index in an event's frames by matching game_clock.

    Parameters
    ----------
    event_frames : list[dict]
        event["frames"], each frame has "game_clock" (seconds remaining in quarter).
    shot_game_clock : float
        Shot time in *game clock seconds remaining in quarter*:
          MINUTES_REMAINING*60 + SECONDS_REMAINING
    fps : int
        Frames per second (25).
    match : str
        - "closest": frame whose game_clock is closest to shot_game_clock
        - "prev": the latest frame with game_clock >= shot_game_clock (i.e., just BEFORE the shot)
          (game_clock counts down)a
    max_time_diff : float
        Maximum allowed absolute time difference (seconds) between shot and matched frame.
        If exceeded, returns None.
    require_shot_clock : bool
        If True, only consider frames that have a non-null shot_clock.
        Useful if some early frames are malformed.

    Returns
    -------
    release_idx : int | None
        Index into event_frames for the best-matching frame, or None if no match within tolerance.
    info : dict
        Debug info: matched_game_clock, dt, num_candidates, etc.
    """

    if event_frames is None or len(event_frames) == 0:
        return None, {"reason": "no_frames"}

    # Extract clocks (and candidate mask)
    clocks = []
    cand_idxs = []
    for i, fr in enumerate(event_frames):
        gc = fr.get("game_clock", None)
        if gc is None:
            continue
        if require_shot_clock and (fr.get("shot_clock", None) is None):
            continue
        clocks.append(float(gc))
        cand_idxs.append(i)

    if len(clocks) == 0:
        return None, {"reason": "no_valid_game_clock_frames"}

    clocks = np.array(clocks, dtype=float)
    cand_idxs = np.array(cand_idxs, dtype=int)

    shot_gc = float(shot_game_clock)

    if match == "closest":
        dt = np.abs(clocks - shot_gc)
        k = int(np.argmin(dt))
        best_idx = int(cand_idxs[k])
        best_dt = float(dt[k])
        best_gc = float(clocks[k])

    elif match == "prev":
        # game_clock counts down; frames just BEFORE the shot have game_clock >= shot_gc
        eligible = np.where(clocks >= shot_gc)[0]
        if len(eligible) == 0:
            # Shot is earlier than all frames; fallback to closest
            dt = np.abs(clocks - shot_gc)
            k = int(np.argmin(dt))
            best_idx = int(cand_idxs[k])
            best_dt = float(dt[k])
            best_gc = float(clocks[k])
            reason = "no_prev_frame_fallback_to_closest"
            if best_dt > max_time_diff:
                return None, {"reason": reason, "best_dt": best_dt, "shot_gc": shot_gc}
            return best_idx, {
                "reason": reason,
                "matched_game_clock": best_gc,
                "dt": best_dt,
                "num_candidates": int(len(clocks)),
            }

        # Among eligible, pick the one with smallest dt (i.e., closest but still before)
        dt_elig = clocks[eligible] - shot_gc
        k = int(np.argmin(dt_elig))
        best_idx = int(cand_idxs[eligible[k]])
        best_dt = float(dt_elig[k])  # non-negative
        best_gc = float(clocks[eligible[k]])

    else:
        raise ValueError("match must be 'closest' or 'prev'")

    if best_dt > max_time_diff:
        return None, {
            "reason": "no_match_within_tolerance",
            "matched_game_clock": best_gc,
            "dt": best_dt,
            "shot_gc": shot_gc,
            "num_candidates": int(len(clocks)),
        }

    return best_idx, {
        "reason": "ok",
        "matched_game_clock": best_gc,
        "dt": best_dt,
        "shot_gc": shot_gc,
        "num_candidates": int(len(clocks)),
    }
