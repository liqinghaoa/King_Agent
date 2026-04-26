#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Baseline feature preprocessor and reward shaping for Gorge Chase SAC.

V4 adds explicit per-episode map memory so the agent can summarize where it
has been, which frontier is still unexplored, and which treasure locations it
has already seen even when they are temporarily out of view.
"""

from collections import deque

import numpy as np

from agent_diy.conf.conf import Config


MAP_SIZE = 128.0
MAP_EDGE = int(MAP_SIZE)
MAP_AREA = MAP_EDGE * MAP_EDGE
LOCAL_VIEW_RADIUS = 10
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
TEMPORAL_POS_DELTA_MAX = 10.0
TEMPORAL_LAST_SEEN_CLIP = 10.0
TEMPORAL_RECENT_VISIBLE_STEPS = 3
TEMPORAL_STREAK_CLIP = 6.0
TEMPORAL_PERSIST_CLIP = 10.0
TEMPORAL_MARGIN_DELTA_MAX = 4.0
TEMPORAL_BRANCH_DELTA_MAX = 4.0
TEMPORAL_DIST_DELTA_MAX = 1.0
TEMPORAL_SUMMARY_FIELDS = [
    "monster_dx",
    "monster_dz",
    "monster_dist_delta",
    "monster_last_seen_steps",
    "monster_recently_visible_flag",
    "encirclement_angle",
    "encirclement_angle_delta",
    "min_margin_delta",
    "dual_side_pressure_flag",
    "hero_dx",
    "hero_dz",
    "same_dir_streak_norm",
    "recent_flash_flag",
    "local_space_delta",
    "margin_delta",
    "danger_rising_flag",
]

SAFE_RESOURCE_DIST_NORM = 0.08
CRITICAL_MONSTER_DIST_NORM = 0.045
LOOP_SURVIVAL_TRIGGER_DIST_NORM = 0.095
LOOP_SURVIVAL_MIN_COVERAGE = 0.30
LOOP_SURVIVAL_MIN_BRANCH_FACTOR = 2
FLASH_CARDINAL_RANGE = 10
FLASH_DIAGONAL_RANGE = 8
FLASH_SOFT_BLOCK_MIN_RATIO = 0.25
FLASH_ESCAPE_MIN_RATIO = 0.45
FLASH_ESCAPE_GAIN_NORM = 0.03
FLASH_DANGER_GAIN_NORM = 0.01
FLASH_ESCAPE_TRIGGER_DIST_NORM = 0.06
FLASH_WASTED_MOVE_NORM = 0.01
LOCAL_DEAD_END_MAX_BRANCH_FACTOR = 1
LOCAL_ESCAPE_MAX_BRANCH_FACTOR = 2
LOCAL_SPACE_SEARCH_DEPTH = 3
LOCAL_SPACE_SCORE_NORMALIZER = 10.0
LOCAL_ESCAPE_SEARCH_DEPTH = 3
CORRIDOR_ESCAPE_SEARCH_DEPTH = 8
CORRIDOR_EXIT_BRANCH_FACTOR = 2
DEAD_END_EXIT_WINDOW_STEPS = 8
ANTI_OSCILLATION_MIN_LOCAL_QUALITY = 0.18
FLASH_COMMIT_STALL_STEPS = 3
LOCAL_COMMIT_TRIGGER_DIST_NORM = 0.08
LOCAL_COMMIT_MIN_QUALITY = 0.20
FRONTIER_FLASH_PATH_MIN_SCORE = 0.45
PLANNER_DANGER_DIST_NORM = FLASH_ESCAPE_TRIGGER_DIST_NORM
PLANNER_SAFE_MARGIN_NORM = 0.10
PLANNER_FLASH_SCORE_DELTA = 2.0
PLANNER_TRUE_THREAT_SCORE_DELTA = 0.5
CHOKE_ESCAPE_OPENNESS_GAIN = 0.18
CHOKE_ESCAPE_BRANCH_GAIN = 2
DEAD_END_SPACE_SCORE = 0.18
DEAD_END_PRETRIGGER_DIST_NORM = 0.10
DEAD_END_PRETRIGGER_PERSIST_STEPS = 2
DEAD_END_COMMIT_HOLD_STEPS = 4
DEAD_END_COMMIT_MAX_HOLD_STEPS = 6
DEAD_END_TARGET_REPLAN_STALL_STEPS = 2
DEAD_END_TARGET_MIN_BRANCH_FACTOR = 2
DEAD_END_TARGET_REACH_DIST_CELLS = 1
DEAD_END_REENTRY_BLOCK_STEPS = 8

MOVE_DELTAS = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]
FOUR_NEIGHBOR_DELTAS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
]

DIR_TO_ACTION = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
}

VEC_TO_ACTION = {
    (1, 0): 0,
    (1, -1): 1,
    (0, -1): 2,
    (-1, -1): 3,
    (-1, 0): 4,
    (-1, 1): 5,
    (0, 1): 6,
    (1, 1): 7,
}


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _safe_exact_dist(hero_pos, unit_pos):
    if not isinstance(hero_pos, dict) or not isinstance(unit_pos, dict):
        return 1.0
    return _norm(
        np.sqrt((hero_pos["x"] - unit_pos["x"]) ** 2 + (hero_pos["z"] - unit_pos["z"]) ** 2),
        MAP_SIZE * 1.41,
    )


def _dir_norm(direction):
    return _norm(direction, 8.0)


def _signed_norm(v, max_abs):
    max_abs = float(max(max_abs, 1e-6))
    return float(np.clip(v, -max_abs, max_abs) / max_abs)


def _dir_delta_norm(current_direction, previous_direction):
    try:
        current_idx = int(current_direction) - 1
        previous_idx = int(previous_direction) - 1
    except (TypeError, ValueError):
        return 0.0
    if current_idx < 0 or previous_idx < 0:
        return 0.0
    delta = ((current_idx - previous_idx + 4) % 8) - 4
    return float(np.clip(delta / 4.0, -1.0, 1.0))


def _angle_norm_from_vectors(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    ax, az = vec_a
    bx, bz = vec_b
    if abs(ax) + abs(az) < 1e-6 or abs(bx) + abs(bz) < 1e-6:
        return 0.0
    angle_a = np.arctan2(float(az), float(ax))
    angle_b = np.arctan2(float(bz), float(bx))
    diff = abs((angle_a - angle_b + np.pi) % (2.0 * np.pi) - np.pi)
    return float(np.clip(diff / np.pi, 0.0, 1.0))


def _empty_monster_track():
    return {
        "has_pos": False,
        "x": 0,
        "z": 0,
        "dist_norm": None,
        "dir_raw": 0,
        "speed_norm": None,
        "last_seen_step": -1000,
        "last_risk": False,
    }


def _new_temporal_stat():
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": 0.0,
        "max": 0.0,
        "nonzero_count": 0,
    }


def _update_temporal_stat(stat_bucket, value):
    value = float(value)
    if not np.isfinite(value):
        value = 0.0
    if stat_bucket["count"] == 0:
        stat_bucket["min"] = value
        stat_bucket["max"] = value
    else:
        stat_bucket["min"] = min(stat_bucket["min"], value)
        stat_bucket["max"] = max(stat_bucket["max"], value)
    stat_bucket["count"] += 1
    stat_bucket["sum"] += value
    stat_bucket["sum_sq"] += value * value
    if abs(value) > 1e-8:
        stat_bucket["nonzero_count"] += 1


def _finalize_temporal_stat(stat_bucket):
    count = int(stat_bucket.get("count", 0))
    if count <= 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "nonzero_rate": 0.0,
        }
    mean = float(stat_bucket["sum"]) / float(count)
    variance = max(0.0, float(stat_bucket["sum_sq"]) / float(count) - mean * mean)
    return {
        "count": count,
        "mean": round(mean, 6),
        "std": round(float(np.sqrt(variance)), 6),
        "min": round(float(stat_bucket["min"]), 6),
        "max": round(float(stat_bucket["max"]), 6),
        "nonzero_rate": round(float(stat_bucket["nonzero_count"]) / float(count), 6),
    }


def _sign(v):
    if abs(v) < 1e-6:
        return 0
    return 1 if v > 0 else -1


def _vector_to_action(dx, dz):
    return VEC_TO_ACTION.get((_sign(dx), _sign(dz)))


def _action_to_dir_norm(action_idx):
    if action_idx is None:
        return 0.0
    return _dir_norm(int(action_idx) + 1)


def _opposite_action(action_idx):
    if action_idx is None:
        return None
    return (int(action_idx) + 4) % 8


def _chebyshev_dist_cells(pos_a, pos_b):
    if pos_a is None or pos_b is None:
        return None
    return max(abs(int(pos_a[0]) - int(pos_b[0])), abs(int(pos_a[1]) - int(pos_b[1])))


def _organ_direction(hero_pos, organ):
    pos = organ.get("pos")
    if isinstance(pos, dict) and "x" in pos and "z" in pos:
        action = _vector_to_action(pos["x"] - hero_pos["x"], pos["z"] - hero_pos["z"])
        if action is not None:
            return action
    return DIR_TO_ACTION.get(int(organ.get("hero_relative_direction", 0)))


def _is_passable(map_info, row, col):
    if map_info is None or len(map_info) == 0:
        return False
    if row < 0 or col < 0 or row >= len(map_info) or col >= len(map_info[0]):
        return False
    return bool(map_info[row][col])


def _can_step(map_info, row, col, next_row, next_col, dx, dz):
    if not _is_passable(map_info, next_row, next_col):
        return False
    if dx != 0 and dz != 0:
        return _is_passable(map_info, row, col + dx) or _is_passable(map_info, row + dz, col)
    return True


def _direction_clear_score(map_info, action_idx, max_steps=4):
    if map_info is None or len(map_info) == 0:
        return 0.5

    center = len(map_info) // 2
    row, col = center, center
    dx, dz = MOVE_DELTAS[action_idx]
    cleared = 0

    for _ in range(max_steps):
        next_row = row + dz
        next_col = col + dx
        if not _can_step(map_info, row, col, next_row, next_col, dx, dz):
            break
        row, col = next_row, next_col
        cleared += 1

    return cleared / max_steps


def _flash_range(action_idx):
    dx, dz = MOVE_DELTAS[action_idx]
    return FLASH_CARDINAL_RANGE if dx == 0 or dz == 0 else FLASH_DIAGONAL_RANGE


def _local_branch_factor(map_info, row, col):
    if not _is_passable(map_info, row, col):
        return 0

    branch_count = 0
    for dx, dz in FOUR_NEIGHBOR_DELTAS:
        if _is_passable(map_info, row + dz, col + dx):
            branch_count += 1
    return branch_count


def _local_space_score(map_info, row, col, max_depth=LOCAL_SPACE_SEARCH_DEPTH):
    if not _is_passable(map_info, row, col):
        return 0.0

    queue = deque([((row, col), 0)])
    visited = {(row, col)}

    while queue:
        (cur_row, cur_col), depth = queue.popleft()
        if depth >= max_depth:
            continue

        for dx, dz in MOVE_DELTAS:
            next_row = cur_row + dz
            next_col = cur_col + dx
            next_key = (next_row, next_col)
            if next_key in visited:
                continue
            if not _can_step(map_info, cur_row, cur_col, next_row, next_col, dx, dz):
                continue
            visited.add(next_key)
            queue.append((next_key, depth + 1))

    return min(1.0, (len(visited) - 1) / LOCAL_SPACE_SCORE_NORMALIZER)


def _local_cell_to_abs_pos(hero_pos, map_info, row, col):
    center = len(map_info) // 2
    return (
        int(hero_pos["x"]) + (col - center),
        int(hero_pos["z"]) + (row - center),
    )


def _visible_monster_positions(monsters):
    positions = []
    for monster in monsters if isinstance(monsters, list) else []:
        if not monster.get("is_in_view", 0):
            continue
        pos = monster.get("pos")
        if isinstance(pos, dict) and "x" in pos and "z" in pos:
            positions.append((int(pos["x"]), int(pos["z"])))
    return positions


def _min_monster_dist_norm_from_pos(monster_positions, pos_x, pos_z, default=1.0):
    if not monster_positions:
        return float(default)
    return min(
        _norm(np.sqrt((pos_x - mx) ** 2 + (pos_z - mz) ** 2), MAP_SIZE * 1.41)
        for mx, mz in monster_positions
    )


def _min_monster_chebyshev_margin_from_pos(monster_positions, pos_x, pos_z, default=4.0):
    if not monster_positions:
        return float(default)
    return float(
        min(max(abs(int(pos_x) - mx), abs(int(pos_z) - mz)) for mx, mz in monster_positions)
    )


def _target_distance_after_move(map_info, hero_pos, action_idx, target_abs_pos):
    if map_info is None or len(map_info) == 0 or target_abs_pos is None:
        return None

    center = len(map_info) // 2
    dx, dz = MOVE_DELTAS[int(action_idx)]
    next_row = center + dz
    next_col = center + dx
    if not _can_step(map_info, center, center, next_row, next_col, dx, dz):
        return None

    next_abs_pos = _local_cell_to_abs_pos(hero_pos, map_info, next_row, next_col)
    return _chebyshev_dist_cells(next_abs_pos, target_abs_pos)


def _get_first_action_towards_abs_target(
    map_info,
    hero_pos,
    target_abs_pos,
    preferred_action=None,
):
    _get_first_action_towards_abs_target.last_path_steps = 0
    _get_first_action_towards_abs_target.last_reachable = False
    if (
        map_info is None
        or len(map_info) == 0
        or target_abs_pos is None
        or hero_pos is None
        or not _in_map_bounds(target_abs_pos[0], target_abs_pos[1])
        or not _in_local_view(hero_pos, target_abs_pos)
    ):
        return None

    center = len(map_info) // 2
    target_col = center + int(target_abs_pos[0]) - int(hero_pos["x"])
    target_row = center + int(target_abs_pos[1]) - int(hero_pos["z"])
    if not _is_passable(map_info, target_row, target_col):
        return None
    if target_row == center and target_col == center:
        _get_first_action_towards_abs_target.last_reachable = True
        return None

    action_order = list(range(8))
    if preferred_action is not None and int(preferred_action) in action_order:
        preferred_action = int(preferred_action)
        action_order = [preferred_action] + [
            action_idx for action_idx in action_order if action_idx != preferred_action
        ]

    queue = deque([((center, center), 0, None)])
    visited = {(center, center)}
    while queue:
        (row, col), steps, first_action = queue.popleft()
        if (row, col) == (target_row, target_col):
            _get_first_action_towards_abs_target.last_path_steps = int(steps)
            _get_first_action_towards_abs_target.last_reachable = True
            return first_action

        for action_idx in action_order:
            dx, dz = MOVE_DELTAS[action_idx]
            next_row = row + dz
            next_col = col + dx
            next_key = (next_row, next_col)
            if next_key in visited:
                continue
            if not _can_step(map_info, row, col, next_row, next_col, dx, dz):
                continue
            visited.add(next_key)
            next_first_action = action_idx if first_action is None else first_action
            queue.append((next_key, steps + 1, next_first_action))

    return None


_get_first_action_towards_abs_target.last_path_steps = 0
_get_first_action_towards_abs_target.last_reachable = False


def _threat_flags_from_pos(pos_x, pos_z, monster_positions, dist_norm=1.0):
    min_margin_cells = _min_monster_chebyshev_margin_from_pos(
        monster_positions,
        pos_x,
        pos_z,
        default=4.0,
    )
    true_threat = min_margin_cells <= 1.0
    near_threat = (not true_threat) and min_margin_cells <= 2.0
    danger = bool(true_threat or near_threat or float(dist_norm) < PLANNER_DANGER_DIST_NORM)
    return true_threat, near_threat, danger, float(min_margin_cells)


def _ray_has_obstacle(map_info, action_idx, landing_step):
    if map_info is None or len(map_info) == 0 or landing_step <= 0:
        return False

    center = len(map_info) // 2
    dx, dz = MOVE_DELTAS[action_idx]
    for step in range(1, int(landing_step) + 1):
        row = center + dz * step
        col = center + dx * step
        if not _is_passable(map_info, row, col):
            return True
    return False


def _estimate_move_action(
    map_info,
    hero_pos,
    monster_positions,
    action_idx,
    current_min_dist_norm,
):
    if map_info is None or len(map_info) == 0:
        return {
            "landing_step": 0,
            "landing_pos": (int(hero_pos["x"]), int(hero_pos["z"])),
            "landing_branch_factor": 0,
            "landing_space_score": 0.0,
            "landing_min_monster_dist_norm": float(current_min_dist_norm),
            "monster_gain": 0.0,
            "expected_move_norm": 0.0,
            "invalid": True,
        }

    center = len(map_info) // 2
    dx, dz = MOVE_DELTAS[action_idx]
    next_row = center + dz
    next_col = center + dx
    landing_step = 1 if _can_step(map_info, center, center, next_row, next_col, dx, dz) else 0
    landing_row = next_row if landing_step > 0 else center
    landing_col = next_col if landing_step > 0 else center

    landing_space_score = (
        _local_space_score(map_info, landing_row, landing_col) if landing_step > 0 else 0.0
    )
    landing_branch_factor = _local_branch_factor(map_info, landing_row, landing_col)
    landing_x, landing_z = _local_cell_to_abs_pos(hero_pos, map_info, landing_row, landing_col)
    landing_min_monster_dist_norm = _min_monster_dist_norm_from_pos(
        monster_positions,
        landing_x,
        landing_z,
        default=current_min_dist_norm,
    )
    monster_gain = landing_min_monster_dist_norm - float(current_min_dist_norm)
    expected_move_norm = _norm(landing_step, MAP_SIZE * 1.41)

    return {
        "landing_step": int(landing_step),
        "landing_row": int(landing_row),
        "landing_col": int(landing_col),
        "landing_pos": (int(landing_x), int(landing_z)),
        "landing_branch_factor": int(landing_branch_factor),
        "landing_space_score": float(landing_space_score),
        "landing_min_monster_dist_norm": float(landing_min_monster_dist_norm),
        "monster_gain": float(monster_gain),
        "expected_move_norm": float(expected_move_norm),
        "invalid": landing_step <= 0,
    }


def _score_escape_candidate(
    candidate,
    hero_pos,
    monster_positions,
    current_true_threat,
    current_near_threat,
    current_danger,
    current_min_dist_norm,
    current_margin_cells,
    current_space_score,
    current_branch_factor,
    action_kind="flash",
):
    landing_pos = candidate.get("landing_pos", (int(hero_pos["x"]), int(hero_pos["z"])))
    landing_min_dist_norm = float(
        candidate.get("landing_min_monster_dist_norm", current_min_dist_norm)
    )
    (
        landing_true_threat,
        landing_near_threat,
        landing_danger,
        landing_margin_cells,
    ) = _threat_flags_from_pos(
        landing_pos[0],
        landing_pos[1],
        monster_positions,
        landing_min_dist_norm,
    )

    distance_gain = landing_min_dist_norm - float(current_min_dist_norm)
    min_margin_gain = float(landing_margin_cells - current_margin_cells)
    openness_gain = float(candidate.get("landing_space_score", 0.0) - current_space_score)
    branch_gain = int(candidate.get("landing_branch_factor", 0) - current_branch_factor)

    leave_danger = bool(current_danger and not landing_danger)
    leave_threat = bool(current_true_threat and not landing_true_threat)
    wall_cross = bool(
        action_kind == "flash"
        and candidate.get("landing_step", 0) > 0
        and candidate.get("path_has_obstacle", False)
        and (leave_danger or distance_gain > 0.0 or openness_gain > 0.0)
    )
    choke_escape = bool(
        current_branch_factor <= LOCAL_ESCAPE_MAX_BRANCH_FACTOR
        and candidate.get("landing_step", 0) > 0
        and (
            openness_gain >= CHOKE_ESCAPE_OPENNESS_GAIN
            or branch_gain >= CHOKE_ESCAPE_BRANCH_GAIN
        )
    )
    dead_end = bool(
        candidate.get("landing_step", 0) > 0
        and candidate.get("landing_branch_factor", 0) <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
        and candidate.get("landing_space_score", 0.0) <= DEAD_END_SPACE_SCORE
    )

    invalid = bool(candidate.get("invalid", False))
    if action_kind == "flash":
        landing_ratio = float(candidate.get("landing_ratio", 0.0))
        if candidate.get("landing_step", 0) <= 0:
            invalid = True
        if landing_pos == (int(hero_pos["x"]), int(hero_pos["z"])):
            invalid = True
        if (
            landing_ratio < FLASH_SOFT_BLOCK_MIN_RATIO
            and not leave_danger
            and distance_gain <= FLASH_DANGER_GAIN_NORM
        ):
            invalid = True
    if not leave_danger and landing_danger and distance_gain < 0.01 and openness_gain <= 0.0:
        invalid = True
    if distance_gain < -0.005:
        invalid = True
    if dead_end and not leave_danger and openness_gain < 0.05:
        invalid = True
    if min_margin_gain < -1.0 and not leave_danger:
        invalid = True

    score = 0.0
    if current_danger:
        score += 5.0 if leave_danger else -4.0
    score += float(np.clip(distance_gain / max(FLASH_ESCAPE_GAIN_NORM, 1e-6), -4.0, 4.0))
    score += float(np.clip(min_margin_gain, -3.0, 3.0))
    score += float(np.clip(openness_gain * 10.0, -3.0, 3.0))
    if wall_cross:
        score += 2.0
    if choke_escape:
        score += 2.0
    if leave_threat:
        score += 2.0
    if dead_end:
        score -= 3.0
    if invalid:
        score -= 6.0

    reason = "CLOSE_ESCAPE"
    if wall_cross:
        reason = "WALL_CROSS_ESCAPE"
    elif choke_escape:
        reason = "CHOKE_ESCAPE"

    candidate.update(
        {
            "current_near_threat": bool(current_near_threat),
            "landing_true_threat": bool(landing_true_threat),
            "landing_near_threat": bool(landing_near_threat),
            "landing_danger": bool(landing_danger),
            "landing_margin_cells": float(landing_margin_cells),
            "distance_gain": float(distance_gain),
            "min_margin_gain": float(min_margin_gain),
            "openness_gain": float(openness_gain),
            "branch_gain": int(branch_gain),
            "leave_danger": bool(leave_danger),
            "leave_threat": bool(leave_threat),
            "wall_cross": bool(wall_cross),
            "choke_escape": bool(choke_escape),
            "dead_end": bool(dead_end),
            "invalid": bool(invalid),
            "score": float(score),
            "reason": reason,
        }
    )
    return candidate


def _estimate_flash_action(
    map_info,
    hero_pos,
    monster_positions,
    action_idx,
    current_min_dist_norm,
):
    if map_info is None or len(map_info) == 0:
        return {
            "landing_step": 0,
            "landing_ratio": 0.0,
            "landing_row": 0,
            "landing_col": 0,
            "landing_pos": (int(hero_pos["x"]), int(hero_pos["z"])),
            "landing_branch_factor": 0,
            "landing_space_score": 0.0,
            "landing_min_monster_dist_norm": float(current_min_dist_norm),
            "monster_gain": 0.0,
            "expected_move_norm": 0.0,
            "quality": -1.0,
            "soft_block": True,
            "path_has_obstacle": False,
            "escape_possible": False,
        }

    center = len(map_info) // 2
    dx, dz = MOVE_DELTAS[action_idx]
    max_steps = _flash_range(action_idx)
    landing_step = 0
    landing_row = center
    landing_col = center

    for step in range(max_steps, 0, -1):
        row = center + dz * step
        col = center + dx * step
        if _is_passable(map_info, row, col):
            landing_step = step
            landing_row = row
            landing_col = col
            break

    landing_ratio = landing_step / float(max_steps)
    landing_branch_factor = _local_branch_factor(map_info, landing_row, landing_col)
    landing_space_score = (
        _local_space_score(map_info, landing_row, landing_col) if landing_step > 0 else 0.0
    )
    landing_x, landing_z = _local_cell_to_abs_pos(hero_pos, map_info, landing_row, landing_col)
    landing_min_monster_dist_norm = _min_monster_dist_norm_from_pos(
        monster_positions,
        landing_x,
        landing_z,
        default=current_min_dist_norm,
    )
    monster_gain = landing_min_monster_dist_norm - float(current_min_dist_norm)
    expected_move_norm = _norm(landing_step, MAP_SIZE * 1.41)
    path_has_obstacle = _ray_has_obstacle(map_info, action_idx, landing_step)

    quality = (
        0.50 * landing_ratio
        + 0.30 * landing_space_score
        + 0.20 * np.clip(monster_gain / max(FLASH_ESCAPE_GAIN_NORM, 1e-6), -1.0, 1.0)
    )
    if landing_step <= 0:
        quality = -1.0

    potential_escape = (
        landing_step > 0
        and landing_ratio >= FLASH_ESCAPE_MIN_RATIO
        and (
            monster_gain >= FLASH_ESCAPE_GAIN_NORM
            or landing_space_score >= 0.55
            or landing_branch_factor >= 2
        )
    )

    soft_block = landing_step <= 0 or (
        landing_ratio < FLASH_SOFT_BLOCK_MIN_RATIO
        and landing_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
        and monster_gain < FLASH_ESCAPE_GAIN_NORM
    )
    if landing_ratio <= 0.25 and monster_gain < 0.02 and not potential_escape:
        soft_block = True
    if current_min_dist_norm > 0.12 and landing_ratio < 0.5 and monster_gain < FLASH_DANGER_GAIN_NORM:
        soft_block = True
    if monster_gain < -0.015 and landing_space_score < 0.3:
        soft_block = True

    escape_possible = potential_escape and not soft_block

    return {
        "landing_step": int(landing_step),
        "landing_ratio": float(landing_ratio),
        "landing_row": int(landing_row),
        "landing_col": int(landing_col),
        "landing_pos": (int(landing_x), int(landing_z)),
        "landing_branch_factor": int(landing_branch_factor),
        "landing_space_score": float(landing_space_score),
        "landing_min_monster_dist_norm": float(landing_min_monster_dist_norm),
        "monster_gain": float(monster_gain),
        "expected_move_norm": float(expected_move_norm),
        "quality": float(quality),
        "soft_block": bool(soft_block),
        "path_has_obstacle": bool(path_has_obstacle),
        "escape_possible": bool(escape_possible),
    }


def _get_local_escape_target(
    map_info,
    hero_pos,
    monster_positions,
    current_min_dist_norm,
    last_move_action=None,
):
    default_meta = {
        "corridor_escape_mode": False,
        "local_escape_is_reverse": False,
        "search_depth": int(LOCAL_ESCAPE_SEARCH_DEPTH),
        "target_branch_factor": 0,
        "target_abs_pos": None,
        "target_path_steps": 0,
        "target_space_score": 0.0,
        "target_monster_gain": 0.0,
        "target_reachable": False,
        "used_exit_candidate": False,
        "used_persistent_target": False,
    }
    _get_local_escape_target.last_meta = dict(default_meta)
    if map_info is None or len(map_info) == 0:
        return 0.0, 0.0, 1.0, None, 0.0

    center = len(map_info) // 2
    current_branch_factor = _local_branch_factor(map_info, center, center)
    corridor_escape_mode = current_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
    search_depth = (
        CORRIDOR_ESCAPE_SEARCH_DEPTH if corridor_escape_mode else LOCAL_ESCAPE_SEARCH_DEPTH
    )
    queue = deque([((center, center), 0, None)])
    visited = {(center, center)}
    best_candidate = None
    best_exit_candidate = None

    while queue:
        (row, col), steps, first_action = queue.popleft()

        if steps > 0 and first_action is not None:
            branch_factor = _local_branch_factor(map_info, row, col)
            space_score = _local_space_score(map_info, row, col)
            abs_x, abs_z = _local_cell_to_abs_pos(hero_pos, map_info, row, col)
            monster_dist_norm = _min_monster_dist_norm_from_pos(
                monster_positions,
                abs_x,
                abs_z,
                default=current_min_dist_norm,
            )
            monster_gain = monster_dist_norm - float(current_min_dist_norm)
            direction_clear = _direction_clear_score(map_info, first_action, max_steps=2)
            local_escape_is_reverse = bool(
                last_move_action is not None
                and first_action == _opposite_action(last_move_action)
            )
            reversal_penalty = 0.0
            if not corridor_escape_mode and local_escape_is_reverse:
                reversal_penalty = 0.08

            quality = (
                0.42 * space_score
                + 0.33 * np.clip(monster_gain / max(FLASH_ESCAPE_GAIN_NORM, 1e-6), -1.0, 1.0)
                + 0.15 * min(branch_factor, 3) / 3.0
                + 0.10 * direction_clear
                - 0.05 * max(0, steps - 1)
                - reversal_penalty
            )
            candidate_key = (
                quality,
                monster_gain,
                space_score,
                branch_factor,
                -steps,
            )
            if best_candidate is None or candidate_key > best_candidate[0]:
                best_candidate = (
                    candidate_key,
                    first_action,
                    steps,
                    quality,
                    branch_factor,
                    local_escape_is_reverse,
                    (int(abs_x), int(abs_z)),
                    float(space_score),
                    float(monster_gain),
                )

            if corridor_escape_mode and branch_factor >= CORRIDOR_EXIT_BRANCH_FACTOR:
                exit_key = (
                    -steps,
                    monster_gain,
                    space_score,
                    branch_factor,
                    direction_clear,
                )
                if best_exit_candidate is None or exit_key > best_exit_candidate[0]:
                    best_exit_candidate = (
                        exit_key,
                        first_action,
                        steps,
                        quality,
                        branch_factor,
                        local_escape_is_reverse,
                        (int(abs_x), int(abs_z)),
                        float(space_score),
                        float(monster_gain),
                    )

        if steps >= search_depth:
            continue

        for action_idx, (dx, dz) in enumerate(MOVE_DELTAS):
            next_row = row + dz
            next_col = col + dx
            next_key = (next_row, next_col)
            if next_key in visited:
                continue
            if not _can_step(map_info, row, col, next_row, next_col, dx, dz):
                continue

            visited.add(next_key)
            next_first_action = action_idx if first_action is None else first_action
            queue.append((next_key, steps + 1, next_first_action))

    chosen_candidate = (
        best_exit_candidate
        if corridor_escape_mode and best_exit_candidate is not None
        else best_candidate
    )
    if chosen_candidate is None:
        _get_local_escape_target.last_meta = {
            "corridor_escape_mode": bool(corridor_escape_mode),
            "local_escape_is_reverse": False,
            "search_depth": int(search_depth),
            "target_branch_factor": 0,
            "target_abs_pos": None,
            "target_path_steps": 0,
            "target_space_score": 0.0,
            "target_monster_gain": 0.0,
            "target_reachable": False,
            "used_exit_candidate": False,
            "used_persistent_target": False,
        }
        return 0.0, 0.0, 1.0, None, 0.0

    (
        _,
        best_action,
        best_steps,
        best_quality,
        target_branch_factor,
        local_escape_is_reverse,
        target_abs_pos,
        target_space_score,
        target_monster_gain,
    ) = chosen_candidate
    _get_local_escape_target.last_meta = {
        "corridor_escape_mode": bool(corridor_escape_mode),
        "local_escape_is_reverse": bool(local_escape_is_reverse),
        "search_depth": int(search_depth),
        "target_branch_factor": int(target_branch_factor),
        "target_abs_pos": target_abs_pos,
        "target_path_steps": int(best_steps),
        "target_space_score": float(target_space_score),
        "target_monster_gain": float(target_monster_gain),
        "target_reachable": True,
        "used_exit_candidate": bool(
            corridor_escape_mode and best_exit_candidate is not None
        ),
        "used_persistent_target": False,
    }
    return (
        1.0,
        _action_to_dir_norm(best_action),
        _norm(best_steps, MAP_SIZE * 1.41),
        best_action,
        float(np.clip(best_quality, 0.0, 1.0)),
    )


_get_local_escape_target.last_meta = {
    "corridor_escape_mode": False,
    "local_escape_is_reverse": False,
    "search_depth": int(LOCAL_ESCAPE_SEARCH_DEPTH),
    "target_branch_factor": 0,
    "target_abs_pos": None,
    "target_path_steps": 0,
    "target_space_score": 0.0,
    "target_monster_gain": 0.0,
    "target_reachable": False,
    "used_exit_candidate": False,
    "used_persistent_target": False,
}


def _detect_two_cell_oscillation(position_history, current_pos):
    if current_pos is None:
        return False

    positions = list(position_history) + [current_pos]
    if len(positions) < 4:
        return False

    last4 = positions[-4:]
    if (
        last4[0] == last4[2]
        and last4[1] == last4[3]
        and last4[0] != last4[1]
    ):
        return True

    return False


def _detect_same_cell_stall(position_history, current_pos, stall_steps=FLASH_COMMIT_STALL_STEPS):
    if current_pos is None:
        return False

    positions = list(position_history) + [current_pos]
    if len(positions) < stall_steps:
        return False

    recent = positions[-stall_steps:]
    return len(set(recent)) == 1


def _adjacent_actions(action_idx):
    action_idx = int(action_idx)
    return ((action_idx - 1) % 8, (action_idx + 1) % 8)


def _in_map_bounds(x, z):
    return 0 <= int(x) < MAP_EDGE and 0 <= int(z) < MAP_EDGE


def _resource_key(sub_type, organ, pos):
    config_id = organ.get("config_id")
    if config_id is not None:
        return (int(sub_type), int(config_id))
    return (int(sub_type), int(pos["x"]), int(pos["z"]))


def _pos_to_dir_norm(hero_pos, target_pos):
    action = _vector_to_action(target_pos[0] - hero_pos["x"], target_pos[1] - hero_pos["z"])
    return _action_to_dir_norm(action)


def _pos_dist_norm(hero_pos, target_pos):
    return _norm(
        np.sqrt((hero_pos["x"] - target_pos[0]) ** 2 + (hero_pos["z"] - target_pos[1]) ** 2),
        MAP_SIZE * 1.41,
    )


def _in_local_view(hero_pos, target_pos):
    return (
        abs(int(target_pos[0]) - int(hero_pos["x"])) <= LOCAL_VIEW_RADIUS
        and abs(int(target_pos[1]) - int(hero_pos["z"])) <= LOCAL_VIEW_RADIUS
    )


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_min_margin_cells = 4.0
        self.last_local_space_score = 0.0
        self.last_local_branch_factor = 0.0
        self.last_danger_state = False
        self.last_true_threat_state = False
        self.last_near_threat_state = False
        self.last_treasure_score = 0.0
        self.last_buff_remain = 0.0
        self.last_hero_pos = None
        self.prev_action_dir = None
        self.same_dir_streak = 0
        self.recent_flash_steps = 0
        self.near_threat_persist_steps = 0
        self.prev_encirclement_angle = 0.0
        self.prev_monster_tracks = [_empty_monster_track() for _ in range(2)]
        self.last_nearest_treasure_dist_norm = 1.0
        self.last_nearest_buff_dist_norm = 1.0
        self.visited_cells = {}
        self.position_history = deque(maxlen=6)

        self.discovered_cells = {}
        self.known_treasures = {}
        self.known_buffs = {}

        self.transition_count = 0
        self.stalled_move_count = 0
        self.oscillation_step_count = 0
        self.revisit_step_count = 0
        self.dead_end_entry_count = 0
        self.discovery_step_count = 0
        self.hidden_treasure_available_steps = 0
        self.frontier_available_step_count = 0
        self.frontier_follow_step_count = 0
        self.loop_survival_mode_step_count = 0
        self.loop_anchor_available_step_count = 0
        self.loop_anchor_follow_step_count = 0
        self.flash_action_count = 0
        self.effective_flash_count = 0
        self.wasted_flash_count = 0
        self.safe_flash_count = 0
        self.danger_flash_count = 0
        self.danger_effective_flash_count = 0
        self.flash_blocked_step_count = 0
        self.flash_eval_trigger_count = 0
        self.flash_execute_count = 0
        self.best_flash_better_than_move_count = 0
        self.no_flash_move_better_count = 0
        self.close_escape_flash_count = 0
        self.wall_cross_flash_count = 0
        self.choke_escape_flash_count = 0
        self.wall_cross_effective_count = 0
        self.choke_escape_effective_count = 0
        self.flash_pre_in_threat_count = 0
        self.flash_pre_in_near_threat_count = 0
        self.flash_leave_danger_count = 0
        self.flash_leave_threat_count = 0
        self.post_flash_dead_end_count = 0
        self.flash_distance_gain_sum = 0.0
        self.flash_min_margin_gain_sum = 0.0
        self.flash_openness_gain_sum = 0.0
        self.dead_end_flash_escape_step_count = 0
        self.dead_end_local_mode_step_count = 0
        self.dead_end_local_commit_step_count = 0
        self.dead_end_flash_available_step_count = 0
        self.dead_end_flash_follow_step_count = 0
        self.dead_end_local_available_step_count = 0
        self.dead_end_local_follow_step_count = 0
        self.dead_end_exit_trigger_count = 0
        self.dead_end_exit_success_count = 0
        self.dead_end_exit_tracking_active = False
        self.dead_end_exit_remaining_steps = 0
        self.dead_end_reverse_available_step_count = 0
        self.dead_end_reverse_follow_step_count = 0
        self.persistent_dead_end_target_active = False
        self.persistent_dead_end_target_pos = None
        self.persistent_dead_end_target_branch_factor = 0
        self.persistent_dead_end_target_steps = 0
        self.persistent_dead_end_commit_remaining = 0
        self.persistent_dead_end_replan_count = 0
        self.persistent_dead_end_follow_available_step_count = 0
        self.persistent_dead_end_follow_step_count = 0
        self.persistent_dead_end_active_step_count = 0
        self.persistent_dead_end_commit_step_count = 0
        self.persistent_dead_end_success_after_follow_count = 0
        self.persistent_dead_end_followed_once = False
        self.dead_end_pretrigger_step_count = 0
        self.dead_end_deeper_block_step_count = 0
        self.confirmed_dead_end_step_count = 0
        self.dead_end_reentry_active_step_count = 0
        self.dead_end_reentry_block_step_count = 0
        self.dead_end_terminal_pos = None
        self.dead_end_reentry_block_steps = 0
        self.persistent_dead_end_active_steps = 0
        self.persistent_dead_end_stall_steps = 0
        self.persistent_dead_end_last_dist = None
        self.nonpersistent_dead_end_commit_remaining = 0
        self.nonpersistent_dead_end_commit_armed = False
        self.last_debug_info = {}

        self.prev_frontier_available = False
        self.prev_frontier_action = None
        self.prev_loop_anchor_available = False
        self.prev_loop_anchor_action = None
        self.prev_loop_survival_mode = False
        self.prev_dead_end_flash_available = False
        self.prev_dead_end_flash_action = None
        self.prev_dead_end_local_available = False
        self.prev_dead_end_local_action = None
        self.prev_dead_end_reverse_available = False
        self.prev_dead_end_reverse_action = None
        self.prev_persistent_dead_end_available = False
        self.prev_persistent_dead_end_action = None
        self.prev_dead_end_pretrigger = False
        self.prev_confirmed_dead_end = False
        self.prev_dead_end_reference_target_pos = None
        self.prev_dead_end_reference_action = None
        self.prev_persistent_dead_end_target_active = False
        self.last_flash_info_by_action = {}
        self.prev_flash_info_by_action = {}
        self.prev_flash_planner_reason = "NO_FLASH_MOVE_IS_BETTER"
        self.temporal_summary_steps = 0
        self.temporal_stats = {
            field_name: _new_temporal_stat() for field_name in TEMPORAL_SUMMARY_FIELDS
        }

    def get_episode_metrics(self):
        steps = max(1, self.transition_count)
        frontier_steps = max(1, self.frontier_available_step_count)
        loop_steps = max(1, self.loop_anchor_available_step_count)
        flash_steps = max(1, self.flash_action_count)
        danger_flash_steps = max(1, self.danger_flash_count)
        planner_steps = max(1, self.flash_eval_trigger_count)
        wall_cross_steps = max(1, self.wall_cross_flash_count)
        dead_end_flash_steps = max(1, self.dead_end_flash_available_step_count)
        dead_end_local_steps = max(1, self.dead_end_local_available_step_count)
        return {
            "stalled_move_rate": round(self.stalled_move_count / steps, 4),
            "oscillation_alert_rate": round(self.oscillation_step_count / steps, 4),
            "effective_flash_rate": round(self.effective_flash_count / flash_steps, 4),
            "wasted_flash_rate": round(self.wasted_flash_count / flash_steps, 4),
            "danger_flash_rate": round(self.danger_flash_count / flash_steps, 4),
            "safe_flash_rate": round(self.safe_flash_count / flash_steps, 4),
            "danger_effective_flash_rate": round(
                self.danger_effective_flash_count / danger_flash_steps,
                4,
            ),
            "flash_eval_trigger_rate": round(self.flash_eval_trigger_count / steps, 4),
            "best_flash_better_than_move_rate": round(
                self.best_flash_better_than_move_count / planner_steps,
                4,
            ),
            "no_flash_move_better_rate": round(
                self.no_flash_move_better_count / planner_steps,
                4,
            ),
            "close_escape_flash_rate": round(self.close_escape_flash_count / flash_steps, 4),
            "wall_cross_flash_rate": round(self.wall_cross_flash_count / flash_steps, 4),
            "choke_escape_flash_rate": round(self.choke_escape_flash_count / flash_steps, 4),
            "wall_cross_effective_rate": round(
                self.wall_cross_effective_count / wall_cross_steps,
                4,
            ),
            "flash_pre_in_threat_rate": round(self.flash_pre_in_threat_count / flash_steps, 4),
            "flash_pre_in_near_threat_rate": round(
                self.flash_pre_in_near_threat_count / flash_steps,
                4,
            ),
            "flash_leave_danger_rate": round(
                self.flash_leave_danger_count / danger_flash_steps,
                4,
            ),
            "flash_leave_threat_rate": round(
                self.flash_leave_threat_count / max(1, self.flash_pre_in_threat_count),
                4,
            ),
            "post_flash_dead_end_rate": round(self.post_flash_dead_end_count / flash_steps, 4),
            "avg_flash_distance_gain": round(self.flash_distance_gain_sum / flash_steps, 4),
            "avg_flash_min_margin_gain": round(
                self.flash_min_margin_gain_sum / flash_steps,
                4,
            ),
            "avg_flash_openness_gain": round(self.flash_openness_gain_sum / flash_steps, 4),
            "flash_blocked_rate": round(self.flash_blocked_step_count / steps, 4),
            "revisit_rate": round(self.revisit_step_count / steps, 4),
            "dead_end_entry_rate": round(self.dead_end_entry_count / steps, 4),
            "dead_end_flash_escape_rate": round(self.dead_end_flash_escape_step_count / steps, 4),
            "dead_end_local_mode_rate": round(self.dead_end_local_mode_step_count / steps, 4),
            "dead_end_local_commit_rate": round(self.dead_end_local_commit_step_count / steps, 4),
            "dead_end_flash_follow_rate": round(
                self.dead_end_flash_follow_step_count / dead_end_flash_steps, 4
            ),
            "dead_end_local_follow_rate": round(
                self.dead_end_local_follow_step_count / dead_end_local_steps, 4
            ),
            "dead_end_exit_success_rate": round(
                self.dead_end_exit_success_count / max(1, self.dead_end_exit_trigger_count),
                4,
            ),
            "dead_end_reverse_follow_rate": round(
                self.dead_end_reverse_follow_step_count
                / max(1, self.dead_end_reverse_available_step_count),
                4,
            ),
            "persistent_dead_end_follow_rate": round(
                self.persistent_dead_end_follow_step_count
                / max(1, self.persistent_dead_end_follow_available_step_count),
                4,
            ),
            "persistent_dead_end_active_rate": round(
                self.persistent_dead_end_active_step_count / steps,
                4,
            ),
            "persistent_dead_end_commit_rate": round(
                self.persistent_dead_end_commit_step_count / steps,
                4,
            ),
            "persistent_dead_end_success_follow_rate": round(
                self.persistent_dead_end_success_after_follow_count
                / max(1, self.dead_end_exit_success_count),
                4,
            ),
            "dead_end_pretrigger_rate": round(
                self.dead_end_pretrigger_step_count / steps,
                4,
            ),
            "dead_end_deeper_block_rate": round(
                self.dead_end_deeper_block_step_count
                / max(
                    1,
                    self.confirmed_dead_end_step_count
                    + self.dead_end_reentry_active_step_count,
                ),
                4,
            ),
            "confirmed_dead_end_rate": round(
                self.confirmed_dead_end_step_count / steps,
                4,
            ),
            "dead_end_reentry_block_rate": round(
                self.dead_end_reentry_block_step_count / steps,
                4,
            ),
            "discovery_step_rate": round(self.discovery_step_count / steps, 4),
            "map_coverage_ratio": round(len(self.discovered_cells) / float(MAP_AREA), 4),
            "hidden_treasure_memory_rate": round(self.hidden_treasure_available_steps / steps, 4),
            "frontier_available_rate": round(self.frontier_available_step_count / steps, 4),
            "frontier_follow_rate": round(self.frontier_follow_step_count / frontier_steps, 4),
            "loop_survival_mode_rate": round(self.loop_survival_mode_step_count / steps, 4),
            "loop_anchor_follow_rate": round(
                self.loop_anchor_follow_step_count / loop_steps, 4
            ),
            "flash_action_count": int(self.flash_action_count),
            "flash_execute_count": int(self.flash_execute_count),
            "effective_flash_count": int(self.effective_flash_count),
            "danger_flash_count": int(self.danger_flash_count),
            "safe_flash_count": int(self.safe_flash_count),
            "danger_effective_flash_count": int(self.danger_effective_flash_count),
            "flash_eval_trigger_count": int(self.flash_eval_trigger_count),
            "flash_leave_danger_count": int(self.flash_leave_danger_count),
            "flash_leave_threat_count": int(self.flash_leave_threat_count),
            "wall_cross_flash_count": int(self.wall_cross_flash_count),
            "wall_cross_effective_count": int(self.wall_cross_effective_count),
            "choke_escape_flash_count": int(self.choke_escape_flash_count),
            "choke_escape_effective_count": int(self.choke_escape_effective_count),
        }

    def _clear_persistent_dead_end_target(self, clear_tracking=False):
        self.persistent_dead_end_target_active = False
        self.persistent_dead_end_target_pos = None
        self.persistent_dead_end_target_branch_factor = 0
        self.persistent_dead_end_target_steps = 0
        self.persistent_dead_end_commit_remaining = 0
        self.persistent_dead_end_replan_count = 0
        self.persistent_dead_end_active_steps = 0
        self.persistent_dead_end_stall_steps = 0
        self.persistent_dead_end_last_dist = None
        self.persistent_dead_end_followed_once = False
        if clear_tracking:
            self.dead_end_exit_tracking_active = False
            self.dead_end_exit_remaining_steps = 0

    def _start_dead_end_reentry_block(self, current_pos_tuple=None):
        terminal_dist = _chebyshev_dist_cells(current_pos_tuple, self.dead_end_terminal_pos)
        if (
            self.dead_end_terminal_pos is not None
            and (
                self.persistent_dead_end_active_steps >= 2
                or self.persistent_dead_end_followed_once
                or (terminal_dist is not None and terminal_dist >= 2)
            )
        ):
            self.dead_end_reentry_block_steps = int(DEAD_END_REENTRY_BLOCK_STEPS)

    def _activate_persistent_dead_end_target(
        self,
        target_abs_pos,
        target_branch_factor,
        target_path_steps,
        replan_count=0,
    ):
        if target_abs_pos is None:
            self._clear_persistent_dead_end_target(clear_tracking=False)
            return False

        was_tracking = bool(self.dead_end_exit_tracking_active)
        self.persistent_dead_end_target_active = True
        self.persistent_dead_end_target_pos = (
            int(target_abs_pos[0]),
            int(target_abs_pos[1]),
        )
        self.persistent_dead_end_target_branch_factor = int(target_branch_factor)
        self.persistent_dead_end_target_steps = int(max(0, target_path_steps or 0))
        self.persistent_dead_end_commit_remaining = int(DEAD_END_COMMIT_HOLD_STEPS)
        self.persistent_dead_end_replan_count = int(max(0, replan_count))
        self.persistent_dead_end_active_steps = 0
        self.persistent_dead_end_stall_steps = 0
        self.persistent_dead_end_last_dist = None
        self.persistent_dead_end_followed_once = False
        if not was_tracking:
            self.dead_end_exit_trigger_count += 1
        self.dead_end_exit_tracking_active = True
        self.dead_end_exit_remaining_steps = int(DEAD_END_EXIT_WINDOW_STEPS)
        return True

    def get_temporal_summary(self):
        summary = {
            "temporal_step_count": int(self.temporal_summary_steps),
            "temporal_feature_dim": int(Config.DIM_OF_TEMPORAL_OBSERVATION),
        }
        for field_name in TEMPORAL_SUMMARY_FIELDS:
            summary[field_name] = _finalize_temporal_stat(
                self.temporal_stats.get(field_name, _new_temporal_stat())
            )
        return summary

    def _record_temporal_stat(self, field_name, value):
        if field_name not in self.temporal_stats:
            self.temporal_stats[field_name] = _new_temporal_stat()
        _update_temporal_stat(self.temporal_stats[field_name], value)

    def _record_temporal_values(
        self,
        hero_dx_norm,
        hero_dz_norm,
        same_dir_streak_norm,
        recent_flash_flag,
        monster_temporal_feats,
        pair_trend_feat,
        risk_summary_feat,
    ):
        self.temporal_summary_steps += 1
        self._record_temporal_stat("hero_dx", hero_dx_norm)
        self._record_temporal_stat("hero_dz", hero_dz_norm)
        self._record_temporal_stat("same_dir_streak_norm", same_dir_streak_norm)
        self._record_temporal_stat("recent_flash_flag", recent_flash_flag)

        for monster_temporal in monster_temporal_feats:
            self._record_temporal_stat("monster_dx", float(monster_temporal[7]))
            self._record_temporal_stat("monster_dz", float(monster_temporal[8]))
            self._record_temporal_stat("monster_dist_delta", float(monster_temporal[9]))
            self._record_temporal_stat("monster_last_seen_steps", float(monster_temporal[12]))
            self._record_temporal_stat(
                "monster_recently_visible_flag",
                float(monster_temporal[13]),
            )

        self._record_temporal_stat("encirclement_angle", float(pair_trend_feat[1]))
        self._record_temporal_stat("encirclement_angle_delta", float(pair_trend_feat[2]))
        self._record_temporal_stat("min_margin_delta", float(pair_trend_feat[3]))
        self._record_temporal_stat("dual_side_pressure_flag", float(pair_trend_feat[4]))
        self._record_temporal_stat("danger_rising_flag", float(risk_summary_feat[11]))
        self._record_temporal_stat("local_space_delta", float(risk_summary_feat[13]))
        self._record_temporal_stat("margin_delta", float(risk_summary_feat[15]))

    def _update_discovered_map(self, hero_pos, map_info):
        if map_info is None or len(map_info) == 0:
            return 0

        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])
        center = len(map_info) // 2
        new_cells = 0

        for row_idx, row in enumerate(map_info):
            for col_idx, cell in enumerate(row):
                abs_x = hero_x + (col_idx - center)
                abs_z = hero_z + (row_idx - center)
                if not _in_map_bounds(abs_x, abs_z):
                    continue
                pos_key = (abs_x, abs_z)
                if pos_key not in self.discovered_cells:
                    new_cells += 1
                self.discovered_cells[pos_key] = bool(cell)

        return new_cells

    def _prune_resource_memory(self, resource_store, hero_pos, visible_keys):
        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])
        to_remove = []

        for key, item in resource_store.items():
            pos_x, pos_z = item["pos"]
            if max(abs(pos_x - hero_x), abs(pos_z - hero_z)) <= LOCAL_VIEW_RADIUS and key not in visible_keys:
                to_remove.append(key)
                continue
            if abs(pos_x - hero_x) <= 1 and abs(pos_z - hero_z) <= 1:
                to_remove.append(key)

        for key in to_remove:
            resource_store.pop(key, None)

    def _update_resource_memory(self, frame_state, hero_pos):
        visible_treasure_keys = set()
        visible_buff_keys = set()

        for organ in frame_state.get("organs", []):
            if organ.get("status", 1) != 1:
                continue

            sub_type = organ.get("sub_type", 0)
            pos = organ.get("pos")
            if not isinstance(pos, dict) or "x" not in pos or "z" not in pos:
                continue

            pos_x = int(pos["x"])
            pos_z = int(pos["z"])
            if not _in_map_bounds(pos_x, pos_z):
                continue

            key = _resource_key(sub_type, organ, pos)
            item = {"pos": (pos_x, pos_z), "last_seen_step": int(self.step_no)}
            if sub_type == 1:
                self.known_treasures[key] = item
                visible_treasure_keys.add(key)
            elif sub_type == 2:
                self.known_buffs[key] = item
                visible_buff_keys.add(key)

        self._prune_resource_memory(self.known_treasures, hero_pos, visible_treasure_keys)
        self._prune_resource_memory(self.known_buffs, hero_pos, visible_buff_keys)

    def _branch_factor(self, pos_key):
        if not self.discovered_cells.get(pos_key, False):
            return 0

        branch_count = 0
        for dx, dz in FOUR_NEIGHBOR_DELTAS:
            next_key = (pos_key[0] + dx, pos_key[1] + dz)
            if self.discovered_cells.get(next_key, False):
                branch_count += 1
        return branch_count

    def _is_frontier(self, pos_key):
        if not self.discovered_cells.get(pos_key, False):
            return False

        for dx, dz in FOUR_NEIGHBOR_DELTAS:
            next_x = pos_key[0] + dx
            next_z = pos_key[1] + dz
            if not _in_map_bounds(next_x, next_z):
                continue
            if (next_x, next_z) not in self.discovered_cells:
                return True
        return False

    def _frontier_unknown_neighbor_count(self, pos_key):
        if not self.discovered_cells.get(pos_key, False):
            return 0

        unknown_count = 0
        for dx, dz in FOUR_NEIGHBOR_DELTAS:
            next_x = pos_key[0] + dx
            next_z = pos_key[1] + dz
            if not _in_map_bounds(next_x, next_z):
                continue
            if (next_x, next_z) not in self.discovered_cells:
                unknown_count += 1
        return unknown_count

    def _can_step_discovered(self, pos_key, next_key, dx, dz):
        if not self.discovered_cells.get(next_key, False):
            return False
        if dx != 0 and dz != 0:
            side_x_key = (pos_key[0] + dx, pos_key[1])
            side_z_key = (pos_key[0], pos_key[1] + dz)
            return self.discovered_cells.get(side_x_key, False) or self.discovered_cells.get(side_z_key, False)
        return True

    def _get_frontier_target(self, hero_pos):
        hero_key = (int(hero_pos["x"]), int(hero_pos["z"]))
        current_is_frontier = self._is_frontier(hero_key)
        if not self.discovered_cells.get(hero_key, False):
            return 0.0, 0.0, 1.0, None, current_is_frontier

        queue = deque([(hero_key, 0, None)])
        visited = {hero_key}
        best_candidate = None
        best_steps = None

        while queue:
            pos_key, steps, first_action = queue.popleft()
            if best_steps is not None and steps > best_steps:
                break

            if pos_key != hero_key and self._is_frontier(pos_key):
                unknown_count = self._frontier_unknown_neighbor_count(pos_key)
                revisit_penalty = min(self.visited_cells.get(pos_key, 0), 9)
                candidate_key = (
                    steps,
                    -unknown_count,
                    revisit_penalty,
                    _pos_dist_norm(hero_pos, pos_key),
                )
                if best_candidate is None or candidate_key < best_candidate[0]:
                    best_candidate = (candidate_key, pos_key, first_action, steps)
                    best_steps = steps
                continue

            if best_steps is not None:
                continue

            for action_idx, (dx, dz) in enumerate(MOVE_DELTAS):
                next_key = (pos_key[0] + dx, pos_key[1] + dz)
                if next_key in visited or not _in_map_bounds(next_key[0], next_key[1]):
                    continue
                if not self._can_step_discovered(pos_key, next_key, dx, dz):
                    continue

                visited.add(next_key)
                next_first_action = action_idx if first_action is None else first_action
                queue.append((next_key, steps + 1, next_first_action))

        if best_candidate is None:
            if current_is_frontier:
                return 1.0, 0.0, 0.0, None, current_is_frontier
            return 0.0, 0.0, 1.0, None, current_is_frontier

        _, best_pos, best_action, best_path_steps = best_candidate
        best_path_dist_norm = _norm(best_path_steps, MAP_SIZE * 1.41)
        return 1.0, _action_to_dir_norm(best_action), best_path_dist_norm, best_action, current_is_frontier

    def _get_known_resource_target(self, hero_pos, resource_store):
        best_pos = None
        best_dist = None
        hidden_flag = 0.0

        for item in resource_store.values():
            pos_key = item["pos"]
            dist_norm = _pos_dist_norm(hero_pos, pos_key)
            if best_dist is None or dist_norm < best_dist:
                best_dist = dist_norm
                best_pos = pos_key

        if best_pos is None:
            return 0.0, 0.0, 1.0, hidden_flag

        if not _in_local_view(hero_pos, best_pos):
            hidden_flag = 1.0

        return 1.0, _pos_to_dir_norm(hero_pos, best_pos), best_dist, hidden_flag

    def _get_loop_anchor_target(self, hero_pos):
        hero_key = (int(hero_pos["x"]), int(hero_pos["z"]))
        if not self.discovered_cells.get(hero_key, False):
            return 0.0, 0.0, 1.0, None

        queue = deque([(hero_key, 0, None)])
        visited = {hero_key}
        best_candidate = None
        best_steps = None

        while queue:
            pos_key, steps, first_action = queue.popleft()
            if best_steps is not None and steps > best_steps:
                break

            if pos_key != hero_key:
                branch_factor = self._branch_factor(pos_key)
                if branch_factor >= 3:
                    revisit_penalty = min(self.visited_cells.get(pos_key, 0), 9)
                    candidate_key = (steps, revisit_penalty, -branch_factor)
                    if best_candidate is None or candidate_key < best_candidate[0]:
                        best_candidate = (candidate_key, pos_key, first_action, steps)
                        best_steps = steps
                    continue

            if best_steps is not None:
                continue

            for action_idx, (dx, dz) in enumerate(MOVE_DELTAS):
                next_key = (pos_key[0] + dx, pos_key[1] + dz)
                if next_key in visited or not _in_map_bounds(next_key[0], next_key[1]):
                    continue
                if not self._can_step_discovered(pos_key, next_key, dx, dz):
                    continue
                visited.add(next_key)
                next_first_action = action_idx if first_action is None else first_action
                queue.append((next_key, steps + 1, next_first_action))

        if best_candidate is None:
            return 0.0, 0.0, 1.0, None

        _, _, best_action, best_path_steps = best_candidate
        best_path_dist_norm = _norm(best_path_steps, MAP_SIZE * 1.41)
        return 1.0, _action_to_dir_norm(best_action), best_path_dist_norm, best_action

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation.get("legal_action", observation.get("legal_act", []))

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        flash_cd = hero.get("flash_cooldown", env_info.get("flash_cooldown", 0))
        buff_remain = hero.get("buff_remaining_time", 0)

        new_discovered_count = self._update_discovered_map(hero_pos, map_info)
        self._update_resource_memory(frame_state, hero_pos)

        hero_feat = np.array(
            [
                _norm(hero_pos["x"], MAP_SIZE),
                _norm(hero_pos["z"], MAP_SIZE),
                _norm(flash_cd, MAX_FLASH_CD),
                _norm(buff_remain, MAX_BUFF_DURATION),
            ],
            dtype=np.float32,
        )

        hero_dx_norm = 0.0
        hero_dz_norm = 0.0
        if self.last_hero_pos is not None:
            hero_dx_norm = _signed_norm(
                int(hero_pos["x"]) - int(self.last_hero_pos["x"]),
                TEMPORAL_POS_DELTA_MAX,
            )
            hero_dz_norm = _signed_norm(
                int(hero_pos["z"]) - int(self.last_hero_pos["z"]),
                TEMPORAL_POS_DELTA_MAX,
            )

        current_action_dir = int(last_action % 8) if last_action >= 0 else None
        current_same_dir_streak = 0
        recent_turn_flag = 0.0
        if current_action_dir is not None:
            if self.prev_action_dir is not None and current_action_dir == self.prev_action_dir:
                current_same_dir_streak = min(self.same_dir_streak + 1, int(TEMPORAL_STREAK_CLIP))
            else:
                current_same_dir_streak = 1
            if self.prev_action_dir is not None and current_action_dir != self.prev_action_dir:
                recent_turn_flag = 1.0
        recent_flash_flag = 1.0 if (last_action >= 8 or self.recent_flash_steps > 0) else 0.0

        monsters = frame_state.get("monsters", [])
        monster_positions = _visible_monster_positions(monsters)
        monster_feats = []
        monster_temporal_feats = []
        effective_monster_tracks = []
        visible_monster_count = 0
        planner_monster_dist_delta = 0.0
        planner_monster_last_seen_steps = 0.0
        planner_monster_recently_visible_flag = 0.0
        planner_monster_priority = None
        for idx in range(2):
            prev_track = dict(self.prev_monster_tracks[idx]) if idx < len(self.prev_monster_tracks) else _empty_monster_track()
            if idx < len(monsters):
                monster = monsters[idx]
                is_in_view = float(monster.get("is_in_view", 0))
                monster_pos = monster["pos"]
                dist_bucket_norm = _norm(monster.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)
                dist_norm = _safe_exact_dist(hero_pos, monster_pos) if is_in_view else dist_bucket_norm
                speed_norm = _norm(monster.get("speed", 1), MAX_MONSTER_SPEED)
                monster_feats.append(
                    np.array(
                        [
                            is_in_view,
                            _norm(monster_pos["x"], MAP_SIZE) if is_in_view else 0.0,
                            _norm(monster_pos["z"], MAP_SIZE) if is_in_view else 0.0,
                            speed_norm,
                            dist_norm,
                            _dir_norm(monster.get("hero_relative_direction", 0)),
                            dist_bucket_norm,
                        ],
                        dtype=np.float32,
                    )
                )

                monster_dx_norm = 0.0
                monster_dz_norm = 0.0
                if is_in_view and prev_track.get("has_pos", False):
                    monster_dx_norm = _signed_norm(
                        int(monster_pos["x"]) - int(prev_track["x"]),
                        TEMPORAL_POS_DELTA_MAX,
                    )
                    monster_dz_norm = _signed_norm(
                        int(monster_pos["z"]) - int(prev_track["z"]),
                        TEMPORAL_POS_DELTA_MAX,
                    )

                prev_dist_norm = prev_track.get("dist_norm")
                monster_dist_delta = (
                    float(np.clip(dist_norm - float(prev_dist_norm), -TEMPORAL_DIST_DELTA_MAX, TEMPORAL_DIST_DELTA_MAX))
                    if prev_dist_norm is not None
                    else 0.0
                )
                monster_dir_delta = _dir_delta_norm(
                    monster.get("hero_relative_direction", 0),
                    prev_track.get("dir_raw", 0),
                )
                prev_speed_norm = prev_track.get("speed_norm")
                monster_speed_delta = (
                    float(np.clip(speed_norm - float(prev_speed_norm), -1.0, 1.0))
                    if prev_speed_norm is not None
                    else 0.0
                )
                if is_in_view:
                    last_seen_steps = 0.0
                    recently_visible_flag = 1.0
                    visible_monster_count += 1
                    track_x = int(monster_pos["x"])
                    track_z = int(monster_pos["z"])
                    has_track_pos = True
                else:
                    last_seen_steps = min(
                        TEMPORAL_LAST_SEEN_CLIP,
                        max(0.0, float(self.step_no - prev_track.get("last_seen_step", -1000))),
                    )
                    recently_visible_flag = float(
                        prev_track.get("has_pos", False)
                        and last_seen_steps <= TEMPORAL_RECENT_VISIBLE_STEPS
                    )
                    track_x = int(prev_track["x"]) if prev_track.get("has_pos", False) else 0
                    track_z = int(prev_track["z"]) if prev_track.get("has_pos", False) else 0
                    has_track_pos = bool(prev_track.get("has_pos", False))

                track_priority = dist_norm if (is_in_view or recently_visible_flag > 0.5) else None
                if track_priority is not None and (
                    planner_monster_priority is None or track_priority < planner_monster_priority
                ):
                    planner_monster_priority = float(track_priority)
                    planner_monster_dist_delta = float(monster_dist_delta)
                    planner_monster_last_seen_steps = _norm(last_seen_steps, TEMPORAL_LAST_SEEN_CLIP)
                    planner_monster_recently_visible_flag = float(recently_visible_flag)

                monster_temporal_feats.append(
                    np.array(
                        [
                            is_in_view,
                            _norm(monster_pos["x"], MAP_SIZE) if is_in_view else 0.0,
                            _norm(monster_pos["z"], MAP_SIZE) if is_in_view else 0.0,
                            speed_norm,
                            dist_norm,
                            _dir_norm(monster.get("hero_relative_direction", 0)),
                            dist_bucket_norm,
                            monster_dx_norm,
                            monster_dz_norm,
                            monster_dist_delta,
                            monster_dir_delta,
                            monster_speed_delta,
                            _norm(last_seen_steps, TEMPORAL_LAST_SEEN_CLIP),
                            recently_visible_flag,
                        ],
                        dtype=np.float32,
                    )
                )
                effective_monster_tracks.append(
                    {
                        "has_pos": has_track_pos,
                        "x": track_x,
                        "z": track_z,
                        "dist_norm": dist_norm if is_in_view or prev_dist_norm is None else float(prev_dist_norm),
                        "dir_raw": int(monster.get("hero_relative_direction", 0)),
                        "speed_norm": speed_norm,
                        "last_seen_step": int(self.step_no) if is_in_view else int(prev_track.get("last_seen_step", -1000)),
                        "last_risk": bool(
                            (dist_norm if is_in_view else (float(prev_dist_norm) if prev_dist_norm is not None else 1.0))
                            < LOOP_SURVIVAL_TRIGGER_DIST_NORM
                        ),
                        "recently_visible": bool(recently_visible_flag),
                        "is_in_view": bool(is_in_view),
                    }
                )
            else:
                monster_feats.append(np.zeros(7, dtype=np.float32))
                last_seen_steps = min(
                    TEMPORAL_LAST_SEEN_CLIP,
                    max(0.0, float(self.step_no - prev_track.get("last_seen_step", -1000))),
                )
                recently_visible_flag = float(
                    prev_track.get("has_pos", False)
                    and last_seen_steps <= TEMPORAL_RECENT_VISIBLE_STEPS
                )
                monster_temporal_feats.append(
                    np.array(
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            _norm(last_seen_steps, TEMPORAL_LAST_SEEN_CLIP),
                            recently_visible_flag,
                        ],
                        dtype=np.float32,
                    )
                )
                effective_monster_tracks.append(
                    {
                        "has_pos": bool(prev_track.get("has_pos", False)),
                        "x": int(prev_track["x"]) if prev_track.get("has_pos", False) else 0,
                        "z": int(prev_track["z"]) if prev_track.get("has_pos", False) else 0,
                        "dist_norm": float(prev_track["dist_norm"]) if prev_track.get("dist_norm") is not None else 1.0,
                        "dir_raw": int(prev_track.get("dir_raw", 0)),
                        "speed_norm": float(prev_track["speed_norm"]) if prev_track.get("speed_norm") is not None else 0.0,
                        "last_seen_step": int(prev_track.get("last_seen_step", -1000)),
                        "last_risk": bool(prev_track.get("last_risk", False)),
                        "recently_visible": bool(recently_visible_flag),
                        "is_in_view": False,
                    }
                )
                track_priority = (
                    float(prev_track["dist_norm"])
                    if prev_track.get("has_pos", False) and recently_visible_flag > 0.5
                    else None
                )
                if track_priority is not None and (
                    planner_monster_priority is None or track_priority < planner_monster_priority
                ):
                    planner_monster_priority = float(track_priority)
                    planner_monster_dist_delta = 0.0
                    planner_monster_last_seen_steps = _norm(last_seen_steps, TEMPORAL_LAST_SEEN_CLIP)
                    planner_monster_recently_visible_flag = float(recently_visible_flag)

        nearest_treasure = [0.0, 0.0, 1.0]
        nearest_buff = [0.0, 0.0, 1.0]
        nearest_treasure_dist_norm = 1.0
        nearest_buff_dist_norm = 1.0

        direction_clear_scores = np.array(
            [_direction_clear_score(map_info, action_idx) for action_idx in range(8)],
            dtype=np.float32,
        )
        treasure_path_scores = np.zeros(8, dtype=np.float32)
        buff_path_scores = np.zeros(8, dtype=np.float32)

        for organ in frame_state.get("organs", []):
            if organ.get("status", 1) != 1:
                continue

            sub_type = organ.get("sub_type", 0)
            rel_dir = _dir_norm(organ.get("hero_relative_direction", 0))
            dist_bucket_norm = _norm(organ.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)
            organ_pos = organ.get("pos")
            if isinstance(organ_pos, dict) and "x" in organ_pos and "z" in organ_pos:
                dist_norm = _safe_exact_dist(hero_pos, organ_pos)
            else:
                dist_norm = dist_bucket_norm

            if sub_type == 1 and dist_norm < nearest_treasure_dist_norm:
                nearest_treasure = [1.0, rel_dir, dist_bucket_norm]
                nearest_treasure_dist_norm = dist_norm
            elif sub_type == 2 and dist_norm < nearest_buff_dist_norm:
                nearest_buff = [1.0, rel_dir, dist_bucket_norm]
                nearest_buff_dist_norm = dist_norm

            action_idx = _organ_direction(hero_pos, organ)
            if action_idx is None:
                continue

            path_signal = max(0.0, 1.0 - dist_norm) * (0.35 + 0.65 * float(direction_clear_scores[action_idx]))
            if sub_type == 1:
                treasure_path_scores[action_idx] = max(treasure_path_scores[action_idx], path_signal)
            elif sub_type == 2:
                buff_weight = 1.0 if buff_remain <= 0 else 0.45
                buff_path_scores[action_idx] = max(buff_path_scores[action_idx], buff_weight * path_signal)

        target_feat = np.array(nearest_treasure + nearest_buff, dtype=np.float32)
        path_feat = np.concatenate([treasure_path_scores, buff_path_scores]).astype(np.float32)

        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for idx in range(min(16, len(legal_act_raw))):
                    legal_action[idx] = int(legal_act_raw[idx])
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < 16}
                legal_action = [1 if idx in valid_set else 0 for idx in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        base_legal_action = list(legal_action)

        hero_pos_key = (int(hero_pos["x"]), int(hero_pos["z"]))
        visit_count = self.visited_cells.get(hero_pos_key, 0) + 1
        self.visited_cells[hero_pos_key] = visit_count

        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, float(m_feat[4]))

        current_local_branch_factor = (
            _local_branch_factor(map_info, len(map_info) // 2, len(map_info) // 2)
            if map_info is not None and len(map_info) > 0
            else 0
        )
        current_local_space_score = (
            _local_space_score(map_info, len(map_info) // 2, len(map_info) // 2)
            if map_info is not None and len(map_info) > 0
            else 0.0
        )
        (
            current_true_threat,
            current_near_threat,
            current_danger,
            current_margin_cells,
        ) = _threat_flags_from_pos(
            int(hero_pos["x"]),
            int(hero_pos["z"]),
            monster_positions,
            cur_min_dist_norm,
        )

        tracked_vectors = []
        tracked_dist_norms = []
        occluded_monster_pressure_flag = 0.0
        for track in effective_monster_tracks:
            if track.get("has_pos", False) and (track.get("is_in_view", False) or track.get("recently_visible", False)):
                tracked_vectors.append(
                    (
                        int(track["x"]) - int(hero_pos["x"]),
                        int(track["z"]) - int(hero_pos["z"]),
                    )
                )
                tracked_dist_norms.append(float(track.get("dist_norm", 1.0)))
            if (
                not track.get("is_in_view", False)
                and track.get("recently_visible", False)
                and (float(track.get("dist_norm", 1.0)) < LOOP_SURVIVAL_TRIGGER_DIST_NORM or track.get("last_risk", False))
            ):
                occluded_monster_pressure_flag = 1.0

        encirclement_angle = (
            _angle_norm_from_vectors(tracked_vectors[0], tracked_vectors[1])
            if len(tracked_vectors) >= 2
            else 0.0
        )
        encirclement_angle_delta = _signed_norm(
            encirclement_angle - float(self.prev_encirclement_angle),
            1.0,
        )
        min_margin_delta = _signed_norm(
            current_margin_cells - self.last_min_margin_cells,
            TEMPORAL_MARGIN_DELTA_MAX,
        )
        dual_side_pressure_flag = float(
            len(tracked_vectors) >= 2
            and encirclement_angle >= 0.35
            and min(tracked_dist_norms) < 0.15
        )
        pair_trend_feat = np.array(
            [
                _norm(visible_monster_count, 2.0),
                float(encirclement_angle),
                float(encirclement_angle_delta),
                float(min_margin_delta),
                float(dual_side_pressure_flag),
            ],
            dtype=np.float32,
        )

        near_threat_persist_steps = (
            min(self.near_threat_persist_steps + 1, int(TEMPORAL_PERSIST_CLIP))
            if (current_near_threat or current_danger)
            else 0
        )
        local_space_delta = _signed_norm(
            current_local_space_score - self.last_local_space_score,
            1.0,
        )
        local_branch_delta = _signed_norm(
            current_local_branch_factor - self.last_local_branch_factor,
            TEMPORAL_BRANCH_DELTA_MAX,
        )
        margin_delta = _signed_norm(
            current_margin_cells - self.last_min_margin_cells,
            TEMPORAL_MARGIN_DELTA_MAX,
        )
        danger_rising_flag = float(
            (current_true_threat and not self.last_true_threat_state)
            or (current_near_threat and not self.last_near_threat_state)
            or (current_danger and not self.last_danger_state)
            or (current_margin_cells < self.last_min_margin_cells - 0.1)
            or (cur_min_dist_norm < self.last_min_monster_dist_norm - 1e-3)
        )

        move_action_infos = {}
        best_move_action = None
        best_move_info = None
        for action_idx in range(8):
            move_info = _estimate_move_action(
                map_info,
                hero_pos,
                monster_positions,
                action_idx,
                cur_min_dist_norm,
            )
            move_info = _score_escape_candidate(
                move_info,
                hero_pos,
                monster_positions,
                current_true_threat,
                current_near_threat,
                current_danger,
                cur_min_dist_norm,
                current_margin_cells,
                current_local_space_score,
                current_local_branch_factor,
                action_kind="move",
            )
            move_action_infos[action_idx] = move_info
            if base_legal_action[action_idx] <= 0:
                continue
            candidate_key = (
                0 if move_info["invalid"] else 1,
                move_info["score"],
                int(move_info["leave_danger"]),
                move_info["distance_gain"],
                move_info["openness_gain"],
            )
            if best_move_info is None or candidate_key > (
                0 if best_move_info["invalid"] else 1,
                best_move_info["score"],
                int(best_move_info["leave_danger"]),
                best_move_info["distance_gain"],
                best_move_info["openness_gain"],
            ):
                best_move_action = action_idx
                best_move_info = move_info

        local_dead_end = current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
        flash_action_infos = {}
        best_flash_action = None
        best_flash_info = None
        blocked_flash_count = 0

        for action_idx in range(8):
            flash_info = _estimate_flash_action(
                map_info,
                hero_pos,
                monster_positions,
                action_idx,
                cur_min_dist_norm,
            )
            flash_info = _score_escape_candidate(
                flash_info,
                hero_pos,
                monster_positions,
                current_true_threat,
                current_near_threat,
                current_danger,
                cur_min_dist_norm,
                current_margin_cells,
                current_local_space_score,
                current_local_branch_factor,
                action_kind="flash",
            )
            flash_action_infos[action_idx] = flash_info
            if base_legal_action[8 + action_idx] <= 0:
                continue
            candidate_key = (
                0 if flash_info["invalid"] else 1,
                flash_info["score"],
                int(flash_info["leave_danger"]),
                flash_info["distance_gain"],
                flash_info["openness_gain"],
            )
            if best_flash_info is None or candidate_key > (
                0 if best_flash_info["invalid"] else 1,
                best_flash_info["score"],
                int(best_flash_info["leave_danger"]),
                best_flash_info["distance_gain"],
                best_flash_info["openness_gain"],
            ):
                best_flash_action = action_idx
                best_flash_info = flash_info

            if flash_info["soft_block"] or flash_info["invalid"]:
                blocked_flash_count += 1

        valid_flash_action_count = sum(
            1
            for action_idx in range(8)
            if base_legal_action[8 + action_idx] > 0
            and not flash_action_infos[action_idx]["invalid"]
        )
        best_move_leaves_danger = bool(best_move_info and best_move_info["leave_danger"])
        best_move_safe = bool(
            best_move_info
            and not best_move_info["invalid"]
            and best_move_info["landing_min_monster_dist_norm"] >= PLANNER_SAFE_MARGIN_NORM
            and best_move_info["landing_space_score"] >= current_local_space_score - 0.05
        )
        flash_eval_trigger = bool(
            flash_cd <= 0
            and valid_flash_action_count > 0
            and current_danger
            and (
                current_true_threat
                or not best_move_leaves_danger
                or not best_move_safe
                or bool(
                    best_flash_info
                    and (
                        best_flash_info["wall_cross"]
                        or best_flash_info["choke_escape"]
                    )
                )
            )
        )
        planner_flash_override = False
        if flash_eval_trigger:
            self.flash_eval_trigger_count += 1
            if (
                best_flash_action is not None
                and best_flash_info is not None
                and not best_flash_info["invalid"]
            ):
                best_move_score = best_move_info["score"] if best_move_info is not None else -1e9
                planner_delta = (
                    PLANNER_TRUE_THREAT_SCORE_DELTA
                    if current_true_threat
                    else PLANNER_FLASH_SCORE_DELTA
                )
                planner_flash_override = bool(
                    (best_flash_info["leave_danger"] and not best_move_leaves_danger)
                    or (best_flash_info["leave_threat"] and current_true_threat)
                    or best_flash_info["score"] >= best_move_score + planner_delta
                )
            if planner_flash_override:
                self.best_flash_better_than_move_count += 1
            else:
                self.no_flash_move_better_count += 1

        legal_flash_action_count = int(valid_flash_action_count)
        narrow_topology_flag = bool(
            current_local_branch_factor <= LOCAL_ESCAPE_MAX_BRANCH_FACTOR
            or blocked_flash_count >= 2
            or legal_flash_action_count <= 3
        )
        dead_end_under_pressure = bool(
            narrow_topology_flag
            and current_danger
        )
        flash_escape_urgent = bool(
            current_true_threat
            or cur_min_dist_norm < FLASH_ESCAPE_TRIGGER_DIST_NORM
            or not best_move_leaves_danger
            or current_local_branch_factor <= LOCAL_ESCAPE_MAX_BRANCH_FACTOR
            or legal_flash_action_count <= 2
            or blocked_flash_count >= 4
        )
        flash_escape_possible = bool(
            dead_end_under_pressure
            and flash_escape_urgent
            and planner_flash_override
            and best_flash_action is not None
            and best_flash_info is not None
        )
        last_move_action = int(last_action % 8) if last_action >= 0 else None
        (
            local_escape_flag,
            local_escape_dir_norm,
            local_escape_dist_norm,
            local_escape_action,
            local_escape_quality,
        ) = _get_local_escape_target(
            map_info,
            hero_pos,
            monster_positions,
            cur_min_dist_norm,
            last_move_action=last_move_action,
        )
        local_escape_meta = dict(getattr(_get_local_escape_target, "last_meta", {}) or {})
        corridor_escape_mode = bool(local_escape_meta.get("corridor_escape_mode", False))
        local_escape_is_reverse = bool(local_escape_meta.get("local_escape_is_reverse", False))
        local_escape_search_depth = int(
            local_escape_meta.get("search_depth", LOCAL_ESCAPE_SEARCH_DEPTH)
        )
        local_escape_target_branch_factor = int(
            local_escape_meta.get("target_branch_factor", 0)
        )
        local_escape_target_abs_pos = local_escape_meta.get("target_abs_pos")
        if (
            isinstance(local_escape_target_abs_pos, (list, tuple))
            and len(local_escape_target_abs_pos) >= 2
        ):
            local_escape_target_abs_pos = (
                int(local_escape_target_abs_pos[0]),
                int(local_escape_target_abs_pos[1]),
            )
        else:
            local_escape_target_abs_pos = None
        planned_local_escape_target_abs_pos = local_escape_target_abs_pos
        local_escape_target_path_steps = int(local_escape_meta.get("target_path_steps", 0))
        planned_local_escape_target_path_steps = int(local_escape_target_path_steps)
        local_escape_target_space_score = float(
            local_escape_meta.get("target_space_score", 0.0)
        )
        local_escape_target_monster_gain = float(
            local_escape_meta.get("target_monster_gain", 0.0)
        )
        local_escape_target_reachable = bool(
            local_escape_meta.get("target_reachable", False)
        )
        current_pos_tuple = (int(hero_pos["x"]), int(hero_pos["z"]))
        dead_end_exit_success = False
        dead_end_deeper_action_blocked = False
        dead_end_reentry_action_blocked = False
        persistent_dead_end_just_activated = False
        persistent_followed_this_step = bool(
            0 <= last_action < 8
            and self.prev_persistent_dead_end_available
            and self.prev_persistent_dead_end_action is not None
            and (last_action % 8) == self.prev_persistent_dead_end_action
        )
        if persistent_followed_this_step:
            self.persistent_dead_end_followed_once = True
        dead_end_pressure_for_pretrigger = bool(
            current_danger
            or danger_rising_flag > 0
            or near_threat_persist_steps >= DEAD_END_PRETRIGGER_PERSIST_STEPS
            or cur_min_dist_norm < DEAD_END_PRETRIGGER_DIST_NORM
        )
        dead_end_pretrigger = bool(
            current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
            and local_escape_action is not None
            and planned_local_escape_target_abs_pos is not None
            and dead_end_pressure_for_pretrigger
        )
        confirmed_dead_end = bool(
            current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
            and local_escape_action is not None
            and planned_local_escape_target_abs_pos is not None
            and local_escape_target_reachable
            and local_escape_target_branch_factor >= CORRIDOR_EXIT_BRANCH_FACTOR
        )
        local_escape_meta["used_persistent_target"] = False
        dead_end_local_mode = bool(
            (confirmed_dead_end or self.persistent_dead_end_target_active)
            and not flash_escape_possible
            and local_escape_action is not None
        )

        def _can_seed_persistent_dead_end_target():
            return bool(
                confirmed_dead_end
                and local_escape_action is not None
                and planned_local_escape_target_abs_pos is not None
            )

        def _activate_current_escape_target(replan_count):
            return self._activate_persistent_dead_end_target(
                planned_local_escape_target_abs_pos,
                local_escape_target_branch_factor,
                planned_local_escape_target_path_steps,
                replan_count=replan_count,
            )

        def _persistent_exit_ready(target_dist):
            terminal_dist = _chebyshev_dist_cells(current_pos_tuple, self.dead_end_terminal_pos)
            has_retreat_progress = bool(
                self.persistent_dead_end_active_steps >= 2
                or self.persistent_dead_end_followed_once
                or (terminal_dist is not None and terminal_dist >= 2)
            )
            if not has_retreat_progress:
                return False
            return bool(
                current_local_branch_factor >= DEAD_END_TARGET_MIN_BRANCH_FACTOR
                or (
                    target_dist is not None
                    and target_dist <= DEAD_END_TARGET_REACH_DIST_CELLS
                )
            )

        if (
            self.persistent_dead_end_target_active
            and self.persistent_dead_end_target_pos is not None
        ):
            current_target_dist = _chebyshev_dist_cells(
                current_pos_tuple,
                self.persistent_dead_end_target_pos,
            )
            if _persistent_exit_ready(current_target_dist):
                dead_end_exit_success = True
                self.dead_end_exit_success_count += 1
                if self.persistent_dead_end_followed_once:
                    self.persistent_dead_end_success_after_follow_count += 1
                self._start_dead_end_reentry_block(current_pos_tuple)
                self._clear_persistent_dead_end_target(clear_tracking=True)
            elif not _in_map_bounds(
                self.persistent_dead_end_target_pos[0],
                self.persistent_dead_end_target_pos[1],
            ):
                self._clear_persistent_dead_end_target(clear_tracking=True)
        elif self.persistent_dead_end_target_active:
            self._clear_persistent_dead_end_target(clear_tracking=True)

        if (
            not dead_end_exit_success
            and not self.persistent_dead_end_target_active
            and _can_seed_persistent_dead_end_target()
        ):
            self.dead_end_terminal_pos = current_pos_tuple
            self.dead_end_reentry_block_steps = 0
            persistent_dead_end_just_activated = bool(
                _activate_current_escape_target(replan_count=0)
            )

        persistent_target_action = None
        persistent_target_reachable = False
        persistent_target_dist = None
        for _ in range(2):
            if (
                not self.persistent_dead_end_target_active
                or self.persistent_dead_end_target_pos is None
            ):
                break

            persistent_target_dist = _chebyshev_dist_cells(
                current_pos_tuple,
                self.persistent_dead_end_target_pos,
            )
            if (
                not persistent_dead_end_just_activated
                and _persistent_exit_ready(persistent_target_dist)
            ):
                dead_end_exit_success = True
                self.dead_end_exit_success_count += 1
                if self.persistent_dead_end_followed_once:
                    self.persistent_dead_end_success_after_follow_count += 1
                self._start_dead_end_reentry_block(current_pos_tuple)
                self._clear_persistent_dead_end_target(clear_tracking=True)
                break

            self.persistent_dead_end_active_steps += 1
            if self.dead_end_exit_tracking_active:
                self.dead_end_exit_remaining_steps = max(
                    0,
                    int(self.dead_end_exit_remaining_steps) - 1,
                )

            persistent_target_action = _get_first_action_towards_abs_target(
                map_info,
                hero_pos,
                self.persistent_dead_end_target_pos,
                preferred_action=local_escape_action,
            )
            persistent_target_reachable = bool(
                getattr(_get_first_action_towards_abs_target, "last_reachable", False)
            )
            persistent_target_path_steps = int(
                getattr(_get_first_action_towards_abs_target, "last_path_steps", 0)
            )
            persistent_commit_progress = bool(
                self.persistent_dead_end_last_dist is None
                or (
                    persistent_target_dist is not None
                    and self.persistent_dead_end_last_dist is not None
                    and persistent_target_dist < self.persistent_dead_end_last_dist
                )
            )

            persistent_target_failed = False
            if persistent_target_action is not None:
                local_escape_action = int(persistent_target_action)
                local_escape_flag = 1.0
                local_escape_dir_norm = _action_to_dir_norm(local_escape_action)
                local_escape_dist_norm = _norm(persistent_target_path_steps, MAP_SIZE * 1.41)
                local_escape_quality = float(np.clip(max(local_escape_quality, 0.45), 0.0, 1.0))
                local_escape_meta["used_persistent_target"] = True
                local_escape_meta["target_abs_pos"] = self.persistent_dead_end_target_pos
                local_escape_meta["target_path_steps"] = int(persistent_target_path_steps)
                local_escape_meta["target_reachable"] = True
                local_escape_meta["target_branch_factor"] = int(
                    self.persistent_dead_end_target_branch_factor
                )
                local_escape_target_abs_pos = self.persistent_dead_end_target_pos
                local_escape_target_path_steps = int(persistent_target_path_steps)
                local_escape_target_reachable = True
                self.persistent_dead_end_target_steps = int(persistent_target_path_steps)
                if persistent_commit_progress:
                    self.persistent_dead_end_stall_steps = 0
                    self.persistent_dead_end_commit_remaining = min(
                        int(DEAD_END_COMMIT_MAX_HOLD_STEPS),
                        max(
                            int(self.persistent_dead_end_commit_remaining),
                            int(DEAD_END_COMMIT_HOLD_STEPS),
                        ),
                    )
                else:
                    self.persistent_dead_end_stall_steps += 1
                    self.persistent_dead_end_commit_remaining = max(
                        0,
                        int(self.persistent_dead_end_commit_remaining) - 1,
                    )
                self.persistent_dead_end_last_dist = persistent_target_dist
            else:
                self.persistent_dead_end_stall_steps += 1
                self.persistent_dead_end_target_steps = 0
                self.persistent_dead_end_commit_remaining = max(
                    0,
                    int(self.persistent_dead_end_commit_remaining) - 1,
                )
                persistent_target_failed = bool(
                    self.persistent_dead_end_stall_steps
                    >= DEAD_END_TARGET_REPLAN_STALL_STEPS
                )

            if self.dead_end_exit_tracking_active and self.dead_end_exit_remaining_steps <= 0:
                persistent_target_failed = True
            if (
                self.persistent_dead_end_commit_remaining <= 0
                and (
                    persistent_target_action is None
                    or self.persistent_dead_end_stall_steps
                    >= DEAD_END_TARGET_REPLAN_STALL_STEPS
                    or _detect_same_cell_stall(
                        self.position_history,
                        current_pos_tuple,
                        stall_steps=2,
                    )
                    or _detect_two_cell_oscillation(
                        self.position_history,
                        current_pos_tuple,
                    )
                )
            ):
                persistent_target_failed = True
            if (
                self.persistent_dead_end_active_steps >= DEAD_END_COMMIT_MAX_HOLD_STEPS
                and (
                    self.persistent_dead_end_stall_steps
                    >= DEAD_END_TARGET_REPLAN_STALL_STEPS
                    or _detect_same_cell_stall(
                        self.position_history,
                        current_pos_tuple,
                        stall_steps=2,
                    )
                    or _detect_two_cell_oscillation(
                        self.position_history,
                        current_pos_tuple,
                    )
                )
            ):
                persistent_target_failed = True

            if not persistent_target_failed:
                break

            next_replan_count = int(self.persistent_dead_end_replan_count) + 1
            if next_replan_count <= 1 and _can_seed_persistent_dead_end_target():
                self.dead_end_terminal_pos = current_pos_tuple
                if _activate_current_escape_target(replan_count=next_replan_count):
                    persistent_target_action = None
                    persistent_target_reachable = False
                    persistent_target_dist = None
                    continue

            self._clear_persistent_dead_end_target(clear_tracking=True)
            persistent_target_action = None
            persistent_target_reachable = False
            persistent_target_dist = None
            break

        dead_end_local_mode = bool(
            (confirmed_dead_end or self.persistent_dead_end_target_active)
            and not flash_escape_possible
            and local_escape_action is not None
        )
        base_local_commit_mode = bool(
            confirmed_dead_end
            and not self.persistent_dead_end_target_active
            and local_escape_action is not None
            and local_escape_quality >= LOCAL_COMMIT_MIN_QUALITY
        )
        if self.persistent_dead_end_target_active or not base_local_commit_mode:
            self.nonpersistent_dead_end_commit_remaining = 0
            self.nonpersistent_dead_end_commit_armed = False
        elif not self.nonpersistent_dead_end_commit_armed:
            self.nonpersistent_dead_end_commit_remaining = 2
            self.nonpersistent_dead_end_commit_armed = True
        elif self.nonpersistent_dead_end_commit_remaining > 0:
            self.nonpersistent_dead_end_commit_remaining = max(
                0,
                int(self.nonpersistent_dead_end_commit_remaining) - 1,
            )
        short_local_commit_mode = bool(
            base_local_commit_mode
            and not self.persistent_dead_end_target_active
            and self.nonpersistent_dead_end_commit_remaining > 0
        )
        persistent_commit_mode = bool(
            self.persistent_dead_end_target_active
            and local_escape_action is not None
            and self.persistent_dead_end_commit_remaining > 0
            and (
                persistent_target_action is not None
                or persistent_dead_end_just_activated
            )
        )
        local_commit_mode = bool(
            persistent_commit_mode or short_local_commit_mode
        )

        flash_commit_mode = bool(
            planner_flash_override
            and best_flash_action is not None
        )
        if flash_commit_mode:
            self.flash_execute_count += 1
        if flash_commit_mode:
            for action_idx in range(8):
                legal_action[action_idx] = 0
                legal_action[8 + action_idx] = 1 if action_idx == best_flash_action else 0
        else:
            # Leave the hard safety gate unchanged in V1. If soft flash
            # candidates are enabled later, this is the place to keep top-K
            # planner-approved flash actions instead of clearing the whole
            # flash branch.
            for action_idx in range(8):
                legal_action[8 + action_idx] = 0
        dead_end_commit_target_pos = (
            self.persistent_dead_end_target_pos
            if self.persistent_dead_end_target_active
            else local_escape_target_abs_pos
        )
        dead_end_hard_commit_mode = bool(
            not flash_commit_mode
            and local_commit_mode
            and local_escape_action is not None
        )
        if dead_end_hard_commit_mode:
            allowed_move_actions = set()
            commit_target_dist = _chebyshev_dist_cells(
                current_pos_tuple,
                dead_end_commit_target_pos,
            )
            primary_action = int(local_escape_action)
            primary_valid = bool(base_legal_action[primary_action] > 0)
            if primary_valid and dead_end_commit_target_pos is not None:
                primary_target_dist = _target_distance_after_move(
                    map_info,
                    hero_pos,
                    primary_action,
                    dead_end_commit_target_pos,
                )
                primary_valid = primary_target_dist is not None and (
                    commit_target_dist is None or primary_target_dist <= commit_target_dist
                )
            if primary_valid:
                allowed_move_actions.add(primary_action)
            else:
                backup_action = None
                backup_key = None
                for action_idx in range(8):
                    if base_legal_action[action_idx] <= 0:
                        continue
                    next_target_dist = _target_distance_after_move(
                        map_info,
                        hero_pos,
                        action_idx,
                        dead_end_commit_target_pos,
                    )
                    if next_target_dist is None:
                        continue
                    if (
                        commit_target_dist is not None
                        and next_target_dist > commit_target_dist
                    ):
                        continue
                    move_score = float(
                        move_action_infos.get(action_idx, {}).get("score", -1e9)
                    )
                    candidate_key = (
                        next_target_dist,
                        -move_score,
                        0 if action_idx == primary_action else 1,
                        action_idx,
                    )
                    if backup_key is None or candidate_key < backup_key:
                        backup_key = candidate_key
                        backup_action = action_idx
                if backup_action is not None:
                    local_escape_action = int(backup_action)
                    local_escape_dir_norm = _action_to_dir_norm(local_escape_action)
                    allowed_move_actions.add(local_escape_action)
            if not allowed_move_actions:
                fallback_action = None
                fallback_key = None
                for action_idx in range(8):
                    if base_legal_action[action_idx] <= 0:
                        continue
                    move_info = move_action_infos.get(action_idx, {})
                    candidate_key = (
                        0 if bool(move_info.get("invalid", False)) else 1,
                        float(move_info.get("score", -1e9)),
                        1 if best_move_action is not None and action_idx == int(best_move_action) else 0,
                        1 if action_idx == primary_action else 0,
                    )
                    if fallback_key is None or candidate_key > fallback_key:
                        fallback_key = candidate_key
                        fallback_action = action_idx
                if fallback_action is not None:
                    local_escape_action = int(fallback_action)
                    local_escape_dir_norm = _action_to_dir_norm(local_escape_action)
                    local_escape_meta["used_persistent_target"] = False
                    allowed_move_actions.add(local_escape_action)
            for action_idx in range(8):
                legal_action[action_idx] = (
                    1 if action_idx in allowed_move_actions and base_legal_action[action_idx] > 0 else 0
                )

        reentry_block_active = bool(
            self.dead_end_reentry_block_steps > 0
            and self.dead_end_terminal_pos is not None
            and not self.persistent_dead_end_target_active
            and not flash_commit_mode
        )
        if reentry_block_active:
            reentry_legal_before_block = list(legal_action[:8])
            current_terminal_dist = _chebyshev_dist_cells(
                current_pos_tuple,
                self.dead_end_terminal_pos,
            )
            blocked_reentry_actions = set()
            for action_idx in range(8):
                if base_legal_action[action_idx] <= 0 or legal_action[action_idx] <= 0:
                    continue
                next_terminal_dist = _target_distance_after_move(
                    map_info,
                    hero_pos,
                    action_idx,
                    self.dead_end_terminal_pos,
                )
                if (
                    next_terminal_dist is not None
                    and current_terminal_dist is not None
                    and next_terminal_dist < current_terminal_dist
                ):
                    blocked_reentry_actions.add(action_idx)
                    legal_action[action_idx] = 0
            if not any(legal_action[:8]):
                for action_idx in range(8):
                    legal_action[action_idx] = reentry_legal_before_block[action_idx]
                blocked_reentry_actions.clear()
            dead_end_reentry_action_blocked = bool(blocked_reentry_actions)
            dead_end_deeper_action_blocked = dead_end_reentry_action_blocked
        if sum(legal_action) == 0:
            fallback_action = None
            if (
                persistent_target_action is not None
                and base_legal_action[int(persistent_target_action)] > 0
            ):
                fallback_action = int(persistent_target_action)
            elif (
                local_escape_action is not None
                and 0 <= int(local_escape_action) < 8
                and base_legal_action[int(local_escape_action)] > 0
            ):
                fallback_action = int(local_escape_action)
            elif best_move_action is not None and base_legal_action[int(best_move_action)] > 0:
                fallback_action = int(best_move_action)
            else:
                for action_idx in range(16):
                    if base_legal_action[action_idx] > 0:
                        fallback_action = action_idx
                        break
            if fallback_action is not None:
                legal_action = [0] * 16
                legal_action[int(fallback_action)] = 1
        if self.dead_end_reentry_block_steps > 0 and not self.persistent_dead_end_target_active:
            self.dead_end_reentry_block_steps = max(
                0,
                int(self.dead_end_reentry_block_steps) - 1,
            )
        legal_flash_action_count = int(sum(legal_action[8:]))

        frontier_flag, frontier_dir_norm, frontier_dist_norm, frontier_action, current_is_frontier = (
            self._get_frontier_target(hero_pos)
        )
        memory_treasure_flag, memory_treasure_dir_norm, memory_treasure_dist_norm, hidden_treasure_flag = (
            self._get_known_resource_target(hero_pos, self.known_treasures)
        )
        loop_flag, loop_dir_norm, loop_dist_norm, loop_action = self._get_loop_anchor_target(hero_pos)
        current_branch_factor = self._branch_factor(hero_pos_key)
        map_coverage_ratio = len(self.discovered_cells) / float(MAP_AREA)
        loop_ready_flag = float(current_branch_factor >= LOOP_SURVIVAL_MIN_BRANCH_FACTOR)
        loop_survival_mode = bool(
            map_coverage_ratio >= LOOP_SURVIVAL_MIN_COVERAGE
            and loop_ready_flag > 0.5
            and loop_flag > 0.5
            and cur_min_dist_norm < LOOP_SURVIVAL_TRIGGER_DIST_NORM
        )

        guidance_flag = frontier_flag
        guidance_dir_norm = frontier_dir_norm
        guidance_dist_norm = frontier_dist_norm
        guidance_source = "frontier"
        if flash_commit_mode and best_flash_action is not None:
            guidance_flag = 1.0
            guidance_dir_norm = _action_to_dir_norm(best_flash_action)
            guidance_dist_norm = 1.0 - float(best_flash_info.get("landing_ratio", 0.0))
            guidance_source = "flash_escape_commit"
        elif local_commit_mode and local_escape_action is not None:
            guidance_flag = local_escape_flag
            guidance_dir_norm = local_escape_dir_norm
            guidance_dist_norm = local_escape_dist_norm
            guidance_source = (
                "persistent_dead_end_commit"
                if self.persistent_dead_end_target_active
                else "dead_end_local_commit"
            )
        elif flash_escape_possible and best_flash_action is not None:
            guidance_flag = 1.0
            guidance_dir_norm = _action_to_dir_norm(best_flash_action)
            guidance_dist_norm = 1.0 - float(best_flash_info.get("landing_ratio", 0.0))
            guidance_source = "flash_planner"
        elif dead_end_local_mode and local_escape_flag > 0.5 and local_escape_action is not None:
            guidance_flag = local_escape_flag
            guidance_dir_norm = local_escape_dir_norm
            guidance_dist_norm = local_escape_dist_norm
            guidance_source = "dead_end_local"
        elif loop_survival_mode and loop_flag > 0.5 and loop_action is not None:
            guidance_flag = loop_flag
            guidance_dir_norm = loop_dir_norm
            guidance_dist_norm = loop_dist_norm
            guidance_source = "loop_survival"

        if guidance_source == "frontier" and cur_min_dist_norm > SAFE_RESOURCE_DIST_NORM:
            for action_idx in range(8):
                if not legal_action[8 + action_idx]:
                    continue
                flash_info = flash_action_infos.get(action_idx, {})
                path_score = max(
                    float(treasure_path_scores[action_idx]),
                    float(buff_path_scores[action_idx]),
                )
                if (
                    float(flash_info.get("monster_gain", 0.0)) <= 0.0
                    and path_score < FRONTIER_FLASH_PATH_MIN_SCORE
                ):
                    legal_action[8 + action_idx] = 0
                    blocked_flash_count += 1
            legal_flash_action_count = int(sum(legal_action[8:]))

        anti_oscillation_mode = bool(
            guidance_source == "frontier"
            and local_escape_action is not None
            and local_escape_quality >= ANTI_OSCILLATION_MIN_LOCAL_QUALITY
            and _detect_two_cell_oscillation(self.position_history, current_pos_tuple)
        )
        if anti_oscillation_mode:
            guidance_flag = local_escape_flag
            guidance_dir_norm = local_escape_dir_norm
            guidance_dist_norm = local_escape_dist_norm
            guidance_source = "anti_oscillation"

        frontier_local_commit_mode = bool(
            guidance_source in {"frontier", "anti_oscillation"}
            and (confirmed_dead_end or self.persistent_dead_end_target_active)
            and not flash_escape_possible
            and local_escape_action is not None
            and local_escape_quality >= LOCAL_COMMIT_MIN_QUALITY
            and cur_min_dist_norm < LOOP_SURVIVAL_TRIGGER_DIST_NORM
            and (
                _detect_same_cell_stall(self.position_history, current_pos_tuple, stall_steps=2)
                or _detect_two_cell_oscillation(self.position_history, current_pos_tuple)
            )
        )
        if frontier_local_commit_mode and not local_commit_mode:
            local_commit_mode = True
            allowed_move_actions = {int(local_escape_action), *_adjacent_actions(local_escape_action)}
            for action_idx in range(8):
                legal_action[action_idx] = 1 if (action_idx in allowed_move_actions and legal_action[action_idx]) else 0
                legal_action[8 + action_idx] = 0
            legal_flash_action_count = 0
            guidance_flag = local_escape_flag
            guidance_dir_norm = local_escape_dir_norm
            guidance_dist_norm = local_escape_dist_norm
            guidance_source = "dead_end_local_commit"

        progress_feat = np.array(
            [
                _norm(self.step_no, self.max_step),
                _norm(min(visit_count, 10), 10.0),
                map_coverage_ratio,
                _norm(current_branch_factor, 4.0),
                guidance_flag,
                guidance_dir_norm,
                guidance_dist_norm,
                memory_treasure_flag,
                memory_treasure_dir_norm,
                memory_treasure_dist_norm,
            ],
            dtype=np.float32,
        )

        hero_temporal_feat = np.array(
            [
                hero_feat[0],
                hero_feat[1],
                hero_feat[2],
                hero_feat[3],
                float(hero_dx_norm),
                float(hero_dz_norm),
                _norm(current_same_dir_streak, TEMPORAL_STREAK_CLIP),
                float(recent_flash_flag),
                float(recent_turn_flag),
            ],
            dtype=np.float32,
        )

        risk_summary_feat = np.array(
            [
                float(current_true_threat),
                float(current_near_threat),
                float(current_danger),
                _norm(min(max(current_margin_cells, 0.0), 4.0), 4.0),
                float(current_local_space_score),
                _norm(min(current_local_branch_factor, 4), 4.0),
                float(dead_end_under_pressure),
                float(flash_escape_urgent),
                float(flash_eval_trigger),
                float(planner_flash_override),
                float(occluded_monster_pressure_flag),
                float(danger_rising_flag),
                _norm(near_threat_persist_steps, TEMPORAL_PERSIST_CLIP),
                float(local_space_delta),
                float(local_branch_delta),
                float(margin_delta),
            ],
            dtype=np.float32,
        )

        same_dir_streak_norm = _norm(current_same_dir_streak, TEMPORAL_STREAK_CLIP)

        last_action_feat = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        if 0 <= int(last_action) < Config.ACTION_NUM:
            last_action_feat[int(last_action)] = 1.0

        temporal_feature = np.concatenate(
            [
                hero_temporal_feat,
                monster_temporal_feats[0],
                monster_temporal_feats[1],
                pair_trend_feat,
                risk_summary_feat,
                last_action_feat,
            ]
        ).astype(np.float32)

        self._record_temporal_values(
            hero_dx_norm=float(hero_dx_norm),
            hero_dz_norm=float(hero_dz_norm),
            same_dir_streak_norm=float(same_dir_streak_norm),
            recent_flash_flag=float(recent_flash_flag),
            monster_temporal_feats=monster_temporal_feats,
            pair_trend_feat=pair_trend_feat,
            risk_summary_feat=risk_summary_feat,
        )

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                target_feat,
                path_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        ).astype(np.float32)

        move_dist_norm = _safe_exact_dist(hero_pos, self.last_hero_pos) if self.last_hero_pos else 0.0
        flash_escape_gain = 0.0
        flash_margin_gain = 0.0
        flash_openness_gain = 0.0
        flash_wasted = False
        flash_effective = False
        flash_danger = False
        prev_flash_info = (
            self.prev_flash_info_by_action.get(last_action % 8, {})
            if last_action >= 8
            else {}
        )

        if last_action >= 8:
            flash_escape_gain = cur_min_dist_norm - self.last_min_monster_dist_norm
            flash_margin_gain = current_margin_cells - self.last_min_margin_cells
            flash_openness_gain = current_local_space_score - self.last_local_space_score
            flash_wasted = (
                move_dist_norm < FLASH_WASTED_MOVE_NORM
                or int(prev_flash_info.get("landing_step", 0)) <= 0
            )
            flash_effective = bool(
                not flash_wasted
                and (
                    flash_escape_gain >= FLASH_ESCAPE_GAIN_NORM
                    or (
                        self.last_min_monster_dist_norm < LOOP_SURVIVAL_TRIGGER_DIST_NORM
                        and move_dist_norm >= 0.02
                        and current_local_branch_factor >= 2
                        and cur_min_dist_norm >= SAFE_RESOURCE_DIST_NORM
                    )
                    or (
                        bool(prev_flash_info.get("escape_possible", False))
                        and move_dist_norm
                        >= 0.6 * float(prev_flash_info.get("expected_move_norm", 0.0))
                    )
                )
            )
            flash_danger = bool(
                cur_min_dist_norm < SAFE_RESOURCE_DIST_NORM
                and current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
                and flash_escape_gain < FLASH_DANGER_GAIN_NORM
            )

        prev_dead_end_target_dist = None
        current_dead_end_target_dist = None
        dead_end_deeper_move_this_step = False
        if self.last_hero_pos is not None and self.prev_dead_end_reference_target_pos is not None:
            prev_dead_end_target_dist = _chebyshev_dist_cells(
                (int(self.last_hero_pos["x"]), int(self.last_hero_pos["z"])),
                self.prev_dead_end_reference_target_pos,
            )
            current_dead_end_target_dist = _chebyshev_dist_cells(
                current_pos_tuple,
                self.prev_dead_end_reference_target_pos,
            )
        if self.last_hero_pos is not None and 0 <= last_action < 8:
            action_dir = last_action % 8
            if (
                self.prev_dead_end_reference_action is not None
                and action_dir == _opposite_action(self.prev_dead_end_reference_action)
            ):
                dead_end_deeper_move_this_step = True
            if (
                prev_dead_end_target_dist is not None
                and current_dead_end_target_dist is not None
                and current_dead_end_target_dist > prev_dead_end_target_dist
            ):
                dead_end_deeper_move_this_step = True

        if self.last_hero_pos is not None:
            self.transition_count += 1
            if move_dist_norm < 0.002:
                self.stalled_move_count += 1
            if anti_oscillation_mode:
                self.oscillation_step_count += 1
            if visit_count > 1:
                self.revisit_step_count += 1
            if current_branch_factor <= 1 and visit_count == 1:
                self.dead_end_entry_count += 1
            if blocked_flash_count > 0:
                self.flash_blocked_step_count += 1
            if flash_escape_possible:
                self.dead_end_flash_escape_step_count += 1
            if dead_end_local_mode:
                self.dead_end_local_mode_step_count += 1
            if confirmed_dead_end:
                self.confirmed_dead_end_step_count += 1
            if self.persistent_dead_end_target_active:
                self.persistent_dead_end_active_step_count += 1
            if reentry_block_active:
                self.dead_end_reentry_active_step_count += 1
            if local_commit_mode:
                self.dead_end_local_commit_step_count += 1
            if persistent_commit_mode:
                self.persistent_dead_end_commit_step_count += 1
            if self.prev_dead_end_flash_available and self.prev_dead_end_flash_action is not None:
                self.dead_end_flash_available_step_count += 1
                if last_action >= 8 and (last_action % 8) == self.prev_dead_end_flash_action:
                    self.dead_end_flash_follow_step_count += 1
            if self.prev_dead_end_local_available and self.prev_dead_end_local_action is not None:
                self.dead_end_local_available_step_count += 1
                if last_action >= 0 and (last_action % 8) == self.prev_dead_end_local_action:
                    self.dead_end_local_follow_step_count += 1
            if self.prev_dead_end_reverse_available and self.prev_dead_end_reverse_action is not None:
                self.dead_end_reverse_available_step_count += 1
                if last_action >= 0 and (last_action % 8) == self.prev_dead_end_reverse_action:
                    self.dead_end_reverse_follow_step_count += 1
            if (
                self.prev_persistent_dead_end_available
                and self.prev_persistent_dead_end_action is not None
            ):
                self.persistent_dead_end_follow_available_step_count += 1
                if (
                    0 <= last_action < 8
                    and (last_action % 8) == self.prev_persistent_dead_end_action
                ):
                    self.persistent_dead_end_follow_step_count += 1
            if dead_end_pretrigger:
                self.dead_end_pretrigger_step_count += 1
            if dead_end_reentry_action_blocked:
                self.dead_end_reentry_block_step_count += 1
                self.dead_end_deeper_block_step_count += 1
            if new_discovered_count > 0 or current_is_frontier:
                self.discovery_step_count += 1
            if hidden_treasure_flag > 0.5:
                self.hidden_treasure_available_steps += 1
            if self.prev_frontier_available:
                self.frontier_available_step_count += 1
                if (
                    last_action >= 0
                    and self.prev_frontier_action is not None
                    and (last_action % 8) == self.prev_frontier_action
                ):
                    self.frontier_follow_step_count += 1
            if self.prev_loop_survival_mode:
                self.loop_survival_mode_step_count += 1
                if self.prev_loop_anchor_available and self.prev_loop_anchor_action is not None:
                    self.loop_anchor_available_step_count += 1
                    if last_action >= 0 and (last_action % 8) == self.prev_loop_anchor_action:
                        self.loop_anchor_follow_step_count += 1
            if last_action >= 8:
                self.flash_action_count += 1
                self.flash_distance_gain_sum += float(flash_escape_gain)
                self.flash_min_margin_gain_sum += float(flash_margin_gain)
                self.flash_openness_gain_sum += float(flash_openness_gain)
                if flash_wasted:
                    self.wasted_flash_count += 1
                elif flash_effective:
                    self.effective_flash_count += 1
                if self.last_danger_state:
                    self.danger_flash_count += 1
                    if flash_effective:
                        self.danger_effective_flash_count += 1
                else:
                    self.safe_flash_count += 1
                if self.last_true_threat_state:
                    self.flash_pre_in_threat_count += 1
                elif self.last_near_threat_state:
                    self.flash_pre_in_near_threat_count += 1
                if self.last_danger_state and not current_danger:
                    self.flash_leave_danger_count += 1
                if self.last_true_threat_state and not current_true_threat:
                    self.flash_leave_threat_count += 1
                if self.prev_flash_planner_reason == "WALL_CROSS_ESCAPE":
                    self.wall_cross_flash_count += 1
                    if flash_effective:
                        self.wall_cross_effective_count += 1
                elif self.prev_flash_planner_reason == "CHOKE_ESCAPE":
                    self.choke_escape_flash_count += 1
                    if flash_effective:
                        self.choke_escape_effective_count += 1
                else:
                    self.close_escape_flash_count += 1
                if (
                    current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
                    and current_local_space_score <= DEAD_END_SPACE_SCORE
                ):
                    self.post_flash_dead_end_count += 1

        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        treasure_score = float(hero.get("treasure_score", env_info.get("treasure_score", 0.0)))
        treasure_reward = 0.02 * max(0.0, treasure_score - self.last_treasure_score)

        safe_for_treasure = cur_min_dist_norm >= SAFE_RESOURCE_DIST_NORM
        if dead_end_under_pressure:
            safe_for_treasure = False
        buff_focus = buff_remain <= 0 or cur_min_dist_norm < SAFE_RESOURCE_DIST_NORM

        treasure_approach_reward = 0.08 * max(
            0.0, self.last_nearest_treasure_dist_norm - nearest_treasure_dist_norm
        )
        if not safe_for_treasure:
            treasure_approach_reward *= 0.25

        buff_approach_reward = 0.06 * max(0.0, self.last_nearest_buff_dist_norm - nearest_buff_dist_norm)
        if not buff_focus:
            buff_approach_reward *= 0.3

        path_follow_reward = 0.0
        if last_action >= 0:
            action_dir = last_action % 8
            if safe_for_treasure:
                path_follow_reward += 0.03 * float(treasure_path_scores[action_dir])
            if buff_focus:
                path_follow_reward += 0.03 * float(buff_path_scores[action_dir])

        buff_reward = 0.25 if buff_remain > self.last_buff_remain + 1e-6 else 0.0
        if visit_count == 1:
            explore_reward = 0.02
            if confirmed_dead_end or self.persistent_dead_end_target_active:
                explore_reward *= 0.25
        else:
            revisit_penalty = -0.006 * min(5, visit_count - 1)
            if loop_survival_mode and loop_flag > 0.5:
                revisit_penalty *= 0.2
            if dead_end_local_mode and local_escape_action is not None:
                revisit_penalty *= 0.1
            explore_reward = revisit_penalty

        if cur_min_dist_norm < CRITICAL_MONSTER_DIST_NORM:
            danger_penalty = -0.12
        elif cur_min_dist_norm < SAFE_RESOURCE_DIST_NORM:
            danger_penalty = -0.04
        else:
            danger_penalty = 0.0

        flash_reward = 0.0
        if last_action >= 8:
            if bool(prev_flash_info.get("soft_block", False)):
                flash_reward -= 0.08
            if flash_effective:
                flash_reward += 0.28 if self.last_min_monster_dist_norm < SAFE_RESOURCE_DIST_NORM else 0.18
            elif flash_wasted:
                flash_reward -= 0.2
            elif float(prev_flash_info.get("landing_ratio", 0.0)) < FLASH_SOFT_BLOCK_MIN_RATIO:
                flash_reward -= 0.08
            elif self.last_min_monster_dist_norm > 0.12:
                flash_reward -= 0.06
            if flash_danger:
                flash_reward -= 0.08

        dead_end_flash_follow_reward = 0.0
        if last_action >= 0 and flash_escape_possible and best_flash_action is not None:
            action_dir = last_action % 8
            if last_action >= 8 and action_dir == best_flash_action:
                dead_end_flash_follow_reward += 0.10 * (
                    0.5
                    + 0.5 * float(best_flash_info.get("landing_ratio", 0.0))
                )
            elif last_action < 8 and cur_min_dist_norm < SAFE_RESOURCE_DIST_NORM:
                dead_end_flash_follow_reward -= 0.03

        loop_follow_reward = 0.0
        if last_action >= 0 and loop_survival_mode and loop_action is not None:
            action_dir = last_action % 8
            if action_dir == loop_action:
                loop_follow_reward += 0.06 * (0.6 + 0.4 * max(0.0, 1.0 - loop_dist_norm))
            elif move_dist_norm < 0.01:
                loop_follow_reward -= 0.08
            elif action_dir == _opposite_action(loop_action):
                loop_follow_reward -= 0.03

        dead_end_local_reward = 0.0
        if last_action >= 0 and dead_end_local_mode and local_escape_action is not None:
            action_dir = last_action % 8
            if action_dir == local_escape_action:
                scale = 0.10 if local_commit_mode else 0.07
                dead_end_local_reward += scale * (0.5 + 0.5 * local_escape_quality)
            elif move_dist_norm < 0.01:
                dead_end_local_reward -= 0.12 if local_commit_mode else 0.10
            elif action_dir == _opposite_action(local_escape_action):
                dead_end_local_reward -= 0.06 if local_commit_mode else 0.04

        dead_end_persistent_follow_reward = 0.0
        if (
            0 <= last_action < 8
            and self.prev_persistent_dead_end_available
            and self.prev_persistent_dead_end_action is not None
            and (last_action % 8) == self.prev_persistent_dead_end_action
        ):
            dead_end_persistent_follow_reward += 0.03

        dead_end_exit_bonus = 0.08 if dead_end_exit_success else 0.0
        dead_end_deeper_penalty = 0.0
        if (
            0 <= last_action < 8
            and (self.prev_confirmed_dead_end or self.prev_persistent_dead_end_target_active)
            and dead_end_deeper_move_this_step
        ):
            dead_end_deeper_penalty -= 0.04

        anti_oscillation_reward = 0.0
        if last_action >= 0 and anti_oscillation_mode and local_escape_action is not None:
            action_dir = last_action % 8
            if action_dir == local_escape_action:
                anti_oscillation_reward += 0.05 * (0.5 + 0.5 * local_escape_quality)
            elif move_dist_norm < 0.01:
                anti_oscillation_reward -= 0.08
            elif action_dir == _opposite_action(local_escape_action):
                anti_oscillation_reward -= 0.05

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_margin_cells = current_margin_cells
        self.last_local_space_score = current_local_space_score
        self.last_local_branch_factor = current_local_branch_factor
        self.last_danger_state = current_danger
        self.last_true_threat_state = current_true_threat
        self.last_near_threat_state = current_near_threat
        self.near_threat_persist_steps = near_threat_persist_steps
        self.prev_encirclement_angle = encirclement_angle
        self.prev_monster_tracks = effective_monster_tracks
        self.last_treasure_score = treasure_score
        self.last_buff_remain = float(buff_remain)
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
        self.same_dir_streak = current_same_dir_streak
        self.prev_action_dir = current_action_dir
        self.recent_flash_steps = 2 if last_action >= 8 else max(0, int(self.recent_flash_steps) - 1)
        self.position_history.append(current_pos_tuple)
        self.last_nearest_treasure_dist_norm = nearest_treasure_dist_norm
        self.last_nearest_buff_dist_norm = nearest_buff_dist_norm
        self.prev_frontier_available = frontier_action is not None
        self.prev_frontier_action = frontier_action
        self.prev_loop_anchor_available = loop_action is not None
        self.prev_loop_anchor_action = loop_action
        self.prev_loop_survival_mode = loop_survival_mode
        self.prev_dead_end_flash_available = flash_escape_possible and best_flash_action is not None
        self.prev_dead_end_flash_action = best_flash_action
        self.prev_dead_end_local_available = dead_end_local_mode and local_escape_action is not None
        self.prev_dead_end_local_action = local_escape_action
        self.prev_dead_end_reverse_available = bool(
            dead_end_local_mode and local_escape_is_reverse and local_escape_action is not None
        )
        self.prev_dead_end_reverse_action = local_escape_action
        self.prev_persistent_dead_end_available = bool(
            self.persistent_dead_end_target_active and local_escape_action is not None
        )
        self.prev_persistent_dead_end_action = (
            int(local_escape_action)
            if self.persistent_dead_end_target_active and local_escape_action is not None
            else None
        )
        self.prev_dead_end_pretrigger = bool(dead_end_pretrigger)
        self.prev_confirmed_dead_end = bool(confirmed_dead_end)
        if self.persistent_dead_end_target_active:
            self.prev_dead_end_reference_target_pos = self.persistent_dead_end_target_pos
        elif confirmed_dead_end:
            self.prev_dead_end_reference_target_pos = local_escape_target_abs_pos
        else:
            self.prev_dead_end_reference_target_pos = None
        self.prev_dead_end_reference_action = (
            int(local_escape_action)
            if (self.persistent_dead_end_target_active or confirmed_dead_end)
            and local_escape_action is not None
            else None
        )
        self.prev_persistent_dead_end_target_active = bool(
            self.persistent_dead_end_target_active
        )
        self.prev_flash_planner_reason = (
            best_flash_info.get("reason", "NO_FLASH_MOVE_IS_BETTER")
            if flash_commit_mode and best_flash_info is not None
            else "NO_FLASH_MOVE_IS_BETTER"
        )
        self.last_flash_info_by_action = flash_action_infos
        self.prev_flash_info_by_action = flash_action_infos
        self.last_debug_info = {
            "step_no": int(self.step_no),
            "hero": [int(hero_pos["x"]), int(hero_pos["z"])],
            "visit_count": int(visit_count),
            "branch_factor": int(current_branch_factor),
            "local_branch_factor": int(current_local_branch_factor),
            "current_local_branch_factor": int(current_local_branch_factor),
            "current_true_threat": int(current_true_threat),
            "current_near_threat": int(current_near_threat),
            "current_danger": int(current_danger),
            "current_margin_cells": round(float(current_margin_cells), 4),
            "current_local_space_score": round(float(current_local_space_score), 4),
            "current_local_branch_delta": round(float(local_branch_delta), 4),
            "current_margin_delta": round(float(margin_delta), 4),
            "current_local_space_delta": round(float(local_space_delta), 4),
            "local_dead_end": int(local_dead_end),
            "narrow_topology_flag": int(narrow_topology_flag),
            "dead_end_under_pressure": int(dead_end_under_pressure),
            "flash_escape_urgent": int(flash_escape_urgent),
            "flash_eval_trigger": int(flash_eval_trigger),
            "flash_planner_override": int(planner_flash_override),
            "occluded_monster_pressure_flag": int(occluded_monster_pressure_flag),
            "danger_rising_flag": int(danger_rising_flag),
            "dead_end_pressure_for_pretrigger": int(dead_end_pressure_for_pretrigger),
            "near_threat_persist_steps": int(near_threat_persist_steps),
            "encirclement_angle": round(float(encirclement_angle), 4),
            "encirclement_angle_delta": round(float(encirclement_angle_delta), 4),
            "dual_side_pressure_flag": int(dual_side_pressure_flag),
            "monster_dist_delta": round(float(planner_monster_dist_delta), 4),
            "monster_last_seen_steps": round(float(planner_monster_last_seen_steps), 4),
            "monster_recently_visible_flag": int(bool(planner_monster_recently_visible_flag)),
            "same_dir_streak_norm": round(float(same_dir_streak_norm), 4),
            "hero_dx": round(float(hero_dx_norm), 4),
            "hero_dz": round(float(hero_dz_norm), 4),
            "recent_flash_flag": int(bool(recent_flash_flag)),
            "flash_commit_mode": int(flash_commit_mode),
            "frontier_local_commit_mode": int(frontier_local_commit_mode),
            "loop_ready_flag": round(loop_ready_flag, 4),
            "loop_survival_mode": int(loop_survival_mode),
            "map_coverage_ratio": round(map_coverage_ratio, 4),
            "frontier_flag": round(float(frontier_flag), 4),
            "frontier_action": int(frontier_action) if frontier_action is not None else -1,
            "frontier_dist_norm": round(float(frontier_dist_norm), 4),
            "loop_flag": round(float(loop_flag), 4),
            "loop_action": int(loop_action) if loop_action is not None else -1,
            "loop_dist_norm": round(float(loop_dist_norm), 4),
            "guidance_source": guidance_source,
            "anti_oscillation_mode": int(anti_oscillation_mode),
            "best_move_action": int(best_move_action) if best_move_action is not None else -1,
            "best_move_score": round(float(best_move_info["score"]), 4) if best_move_info else 0.0,
            "best_move_leave_danger": int(bool(best_move_info and best_move_info["leave_danger"])),
            "best_move_dist_gain": round(float(best_move_info["distance_gain"]), 4)
            if best_move_info
            else 0.0,
            "memory_treasure_flag": round(float(memory_treasure_flag), 4),
            "memory_treasure_dir_norm": round(float(memory_treasure_dir_norm), 4),
            "memory_treasure_dist_norm": round(float(memory_treasure_dist_norm), 4),
            "hidden_treasure_flag": round(float(hidden_treasure_flag), 4),
            "nearest_treasure_dist_norm": round(float(nearest_treasure_dist_norm), 4),
            "nearest_buff_dist_norm": round(float(nearest_buff_dist_norm), 4),
            "cur_min_monster_dist_norm": round(float(cur_min_dist_norm), 4),
            "flash_escape_possible": int(flash_escape_possible),
            "dead_end_local_mode": int(dead_end_local_mode),
            "confirmed_dead_end": int(confirmed_dead_end),
            "dead_end_pretrigger": int(dead_end_pretrigger),
            "dead_end_local_commit_mode": int(local_commit_mode),
            "corridor_escape_mode": int(corridor_escape_mode),
            "local_escape_is_reverse": int(local_escape_is_reverse),
            "local_escape_search_depth": int(local_escape_search_depth),
            "local_escape_target_branch_factor": int(local_escape_target_branch_factor),
            "local_escape_target_abs_pos": (
                [int(local_escape_target_abs_pos[0]), int(local_escape_target_abs_pos[1])]
                if local_escape_target_abs_pos is not None
                else [-1, -1]
            ),
            "local_escape_target_path_steps": int(local_escape_target_path_steps),
            "local_escape_target_space_score": round(float(local_escape_target_space_score), 4),
            "local_escape_target_monster_gain": round(float(local_escape_target_monster_gain), 4),
            "local_escape_target_reachable": int(local_escape_target_reachable),
            "dead_end_exit_tracking_active": int(self.dead_end_exit_tracking_active),
            "dead_end_exit_remaining_steps": int(self.dead_end_exit_remaining_steps),
            "dead_end_exit_success": int(dead_end_exit_success),
            "dead_end_terminal_pos": (
                [int(self.dead_end_terminal_pos[0]), int(self.dead_end_terminal_pos[1])]
                if self.dead_end_terminal_pos is not None
                else [-1, -1]
            ),
            "dead_end_reentry_block_steps": int(self.dead_end_reentry_block_steps),
            "dead_end_reentry_action_blocked": int(dead_end_reentry_action_blocked),
            "persistent_dead_end_target_active": int(self.persistent_dead_end_target_active),
            "persistent_dead_end_target_pos": (
                [
                    int(self.persistent_dead_end_target_pos[0]),
                    int(self.persistent_dead_end_target_pos[1]),
                ]
                if self.persistent_dead_end_target_pos is not None
                else [-1, -1]
            ),
            "persistent_dead_end_target_branch_factor": int(
                self.persistent_dead_end_target_branch_factor
            ),
            "persistent_dead_end_active_steps": int(self.persistent_dead_end_active_steps),
            "persistent_dead_end_target_steps": int(self.persistent_dead_end_target_steps),
            "persistent_dead_end_commit_remaining": int(
                self.persistent_dead_end_commit_remaining
            ),
            "persistent_dead_end_replan_count": int(
                self.persistent_dead_end_replan_count
            ),
            "persistent_dead_end_followed_once": int(self.persistent_dead_end_followed_once),
            "persistent_dead_end_just_activated": int(persistent_dead_end_just_activated),
            "persistent_dead_end_commit_mode": int(persistent_commit_mode),
            "persistent_dead_end_action": (
                int(local_escape_action)
                if self.persistent_dead_end_target_active and local_escape_action is not None
                else -1
            ),
            "nonpersistent_dead_end_commit_remaining": int(
                self.nonpersistent_dead_end_commit_remaining
            ),
            "used_persistent_target": int(
                bool(local_escape_meta.get("used_persistent_target", False))
            ),
            "dead_end_deeper_action_blocked": int(dead_end_deeper_action_blocked),
            "dead_end_flash_action": int(best_flash_action) if flash_escape_possible and best_flash_action is not None else -1,
            "local_escape_action": int(local_escape_action) if local_escape_action is not None else -1,
            "local_escape_quality": round(float(local_escape_quality), 4),
            "local_escape_dist_norm": round(float(local_escape_dist_norm), 4),
            "blocked_flash_count": int(blocked_flash_count),
            "best_flash_action": int(best_flash_action) if best_flash_action is not None else -1,
            "best_flash_score": round(float(best_flash_info["score"]), 4) if best_flash_info else 0.0,
            "best_flash_reason": best_flash_info.get("reason", "NO_FLASH_NO_SAFE_LANDING")
            if best_flash_info
            else "NO_FLASH_NO_SAFE_LANDING",
            "best_flash_leave_danger": int(bool(best_flash_info and best_flash_info["leave_danger"])),
            "best_flash_leave_threat": int(bool(best_flash_info and best_flash_info["leave_threat"])),
            "best_flash_wall_cross": int(bool(best_flash_info and best_flash_info["wall_cross"])),
            "best_flash_choke_escape": int(bool(best_flash_info and best_flash_info["choke_escape"])),
            "best_flash_distance_gain": round(float(best_flash_info["distance_gain"]), 4)
            if best_flash_info
            else 0.0,
            "best_flash_min_margin_gain": round(float(best_flash_info["min_margin_gain"]), 4)
            if best_flash_info
            else 0.0,
            "best_flash_openness_gain": round(float(best_flash_info["openness_gain"]), 4)
            if best_flash_info
            else 0.0,
            "best_flash_landing_ratio": round(
                float(best_flash_info["landing_ratio"]), 4
            )
            if best_flash_info
            else 0.0,
            "best_flash_space_score": round(
                float(best_flash_info["landing_space_score"]), 4
            )
            if best_flash_info
            else 0.0,
            "best_flash_monster_gain": round(float(best_flash_info["monster_gain"]), 4)
            if best_flash_info
            else 0.0,
            "legal_action_count": int(sum(legal_action)),
            "legal_flash_action_count": legal_flash_action_count,
        }

        reward = [
            survive_reward
            + dist_shaping
            + treasure_reward
            + treasure_approach_reward
            + buff_reward
            + buff_approach_reward
            + path_follow_reward
            + explore_reward
            + danger_penalty
            + flash_reward
            + dead_end_flash_follow_reward
            + loop_follow_reward
            + dead_end_local_reward
            + dead_end_persistent_follow_reward
            + dead_end_exit_bonus
            + dead_end_deeper_penalty
            + anti_oscillation_reward
        ]

        return feature, temporal_feature, legal_action, reward
