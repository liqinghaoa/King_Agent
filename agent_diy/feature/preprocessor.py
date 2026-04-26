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
SOFT_FLASH_KEEP_TOPK = 2
SOFT_FLASH_KEEP_SCORE_TOLERANCE = 1.0
LOCAL_DEAD_END_MAX_BRANCH_FACTOR = 1
LOCAL_ESCAPE_MAX_BRANCH_FACTOR = 2
LOCAL_SPACE_SEARCH_DEPTH = 3
LOCAL_SPACE_SCORE_NORMALIZER = 10.0
LOCAL_ESCAPE_SEARCH_DEPTH = 3
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
DEAD_END_BACKTRACK_GAIN_TOLERANCE = 0.01
DEAD_END_BACKTRACK_MIN_DIST_NORM = 0.05
BACKTRACK_EXIT_MIN_ROUTE_STEPS = 2
LOCAL_REACTION_MIN_HISTORY = 4
EARLY_TACTICAL_GRACE_STEPS = 6
POST_FLASH_FOLLOW_MIN_DIST_NORM = 0.004
POST_FLASH_PAUSE_DIST_NORM = 0.002
ANTI_OSCILLATION_GOAL_NEAR_DIST_NORM = 0.04
BACKTRACK_RETRY_SAFE_DIST_NORM = 0.10
HIGH_PRESSURE_SURVIVE_REWARD = 0.006
TREASURE_SCORE_REWARD_SCALE = 0.03
TREASURE_APPROACH_REWARD_SCALE = 0.12
UNSAFE_TREASURE_APPROACH_SCALE = 0.35
PRESSURE_TREASURE_WINDOW_DIST_NORM = 0.12
PRESSURE_TREASURE_MIN_SPACE_SCORE = 0.22
BUFF_APPROACH_REWARD_SCALE = 0.05
PASSIVE_BUFF_APPROACH_SCALE = 0.25
TREASURE_PATH_FOLLOW_REWARD = 0.05
BUFF_PATH_FOLLOW_REWARD = 0.02
EXPLORE_REWARD_ON_NEW_CELL = 0.01
REVISIT_PENALTY_SCALE = 0.007
LOOP_REVISIT_PENALTY_DISCOUNT = 0.35
BACKTRACK_REVISIT_PENALTY_DISCOUNT = 0.65
DEAD_END_LOCAL_REVISIT_PENALTY_DISCOUNT = 0.15

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


def _abs_pos_to_local_cell(hero_pos, map_info, pos_key):
    if map_info is None or len(map_info) == 0 or pos_key is None:
        return None

    center = len(map_info) // 2
    col = center + int(pos_key[0]) - int(hero_pos["x"])
    row = center + int(pos_key[1]) - int(hero_pos["z"])
    if row < 0 or row >= len(map_info) or col < 0 or col >= len(map_info[0]):
        return None
    return row, col


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
    if map_info is None or len(map_info) == 0:
        return 0.0, 0.0, 1.0, None, 0.0

    center = len(map_info) // 2
    queue = deque([((center, center), 0, None)])
    visited = {(center, center)}
    best_candidate = None

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
            reversal_penalty = 0.0
            if last_move_action is not None and first_action == _opposite_action(last_move_action):
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
                best_candidate = (candidate_key, first_action, steps, quality)

        if steps >= LOCAL_ESCAPE_SEARCH_DEPTH:
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

    if best_candidate is None:
        return 0.0, 0.0, 1.0, None, 0.0

    _, best_action, best_steps, best_quality = best_candidate
    return (
        1.0,
        _action_to_dir_norm(best_action),
        _norm(best_steps, MAP_SIZE * 1.41),
        best_action,
        float(np.clip(best_quality, 0.0, 1.0)),
    )


def _get_backtrack_exit_target(
    route_history,
    hero_pos,
    map_info,
    monster_positions,
    current_min_dist_norm,
    current_local_branch_factor,
):
    if map_info is None or len(map_info) == 0:
        return 0.0, 0.0, 1.0, None, 0.0, 0, None, {
            "route_steps": 0,
            "branch_factor": 0,
            "space_score": 0.0,
            "monster_dist_norm": float(current_min_dist_norm),
            "monster_gain": 0.0,
            "quality": -1.0,
        }

    route_positions = list(route_history)
    if len(route_positions) < 2:
        return 0.0, 0.0, 1.0, None, 0.0, 0, None, {
            "route_steps": 0,
            "branch_factor": 0,
            "space_score": 0.0,
            "monster_dist_norm": float(current_min_dist_norm),
            "monster_gain": 0.0,
            "quality": -1.0,
        }

    current_pos = route_positions[-1]
    previous_route = route_positions[:-1]
    first_action = None
    best_candidate = None
    seen_positions = set()

    for route_steps, pos_key in enumerate(reversed(previous_route), start=1):
        if pos_key == current_pos or pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        if first_action is None:
            first_action = _vector_to_action(
                pos_key[0] - current_pos[0],
                pos_key[1] - current_pos[1],
            )
        if first_action is None:
            break

        local_cell = _abs_pos_to_local_cell(hero_pos, map_info, pos_key)
        if local_cell is None:
            continue
        row, col = local_cell
        if not _is_passable(map_info, row, col):
            continue

        branch_factor = _local_branch_factor(map_info, row, col)
        space_score = _local_space_score(map_info, row, col)
        monster_dist_norm = _min_monster_dist_norm_from_pos(
            monster_positions,
            int(pos_key[0]),
            int(pos_key[1]),
            default=current_min_dist_norm,
        )
        monster_gain = monster_dist_norm - float(current_min_dist_norm)
        route_progress = min(route_steps, 8) / 8.0
        exit_bonus = 0.0
        if branch_factor >= 2:
            exit_bonus += 0.10
        if branch_factor > current_local_branch_factor:
            exit_bonus += 0.08
        if space_score >= 0.45:
            exit_bonus += 0.07

        quality = (
            0.34 * space_score
            + 0.28 * np.clip(monster_gain / max(FLASH_ESCAPE_GAIN_NORM, 1e-6), -1.0, 1.0)
            + 0.18 * min(branch_factor, 3) / 3.0
            + 0.12 * route_progress
            + exit_bonus
        )
        candidate_key = (
            quality,
            branch_factor,
            space_score,
            monster_gain,
            route_progress,
        )
        if best_candidate is None or candidate_key > best_candidate[0]:
            best_candidate = (
                candidate_key,
                first_action,
                route_steps,
                pos_key,
                {
                    "route_steps": int(route_steps),
                    "branch_factor": int(branch_factor),
                    "space_score": float(space_score),
                    "monster_dist_norm": float(monster_dist_norm),
                    "monster_gain": float(monster_gain),
                    "quality": float(quality),
                },
            )

    if best_candidate is None:
        return 0.0, 0.0, 1.0, None, 0.0, 0, None, {
            "route_steps": 0,
            "branch_factor": 0,
            "space_score": 0.0,
            "monster_dist_norm": float(current_min_dist_norm),
            "monster_gain": 0.0,
            "quality": -1.0,
        }

    _, backtrack_action, route_steps, exit_pos, exit_info = best_candidate
    return (
        1.0,
        _action_to_dir_norm(backtrack_action),
        _norm(route_steps, MAP_SIZE * 1.41),
        backtrack_action,
        float(np.clip(exit_info["quality"], 0.0, 1.0)),
        int(route_steps),
        exit_pos,
        exit_info,
    )


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
        self.last_danger_state = False
        self.last_true_threat_state = False
        self.last_near_threat_state = False
        self.last_treasure_score = 0.0
        self.last_buff_remain = 0.0
        self.last_hero_pos = None
        self.last_nearest_treasure_dist_norm = 1.0
        self.last_nearest_buff_dist_norm = 1.0
        self.visited_cells = {}
        self.position_history = deque(maxlen=6)
        self.route_history = deque(maxlen=64)

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
        self.best_flash_better_than_move_count = 0
        self.no_flash_move_better_count = 0
        self.close_escape_flash_count = 0
        self.wall_cross_flash_count = 0
        self.choke_escape_flash_count = 0
        self.wall_cross_effective_count = 0
        self.flash_pre_in_threat_count = 0
        self.flash_pre_in_near_threat_count = 0
        self.flash_leave_danger_count = 0
        self.flash_leave_threat_count = 0
        self.post_flash_dead_end_count = 0
        self.flash_distance_gain_sum = 0.0
        self.flash_min_margin_gain_sum = 0.0
        self.flash_openness_gain_sum = 0.0
        self.dead_end_flash_escape_step_count = 0
        self.dead_end_backtrack_step_count = 0
        self.dead_end_local_mode_step_count = 0
        self.dead_end_local_commit_step_count = 0
        self.dead_end_flash_available_step_count = 0
        self.dead_end_flash_follow_step_count = 0
        self.dead_end_backtrack_available_step_count = 0
        self.dead_end_backtrack_follow_step_count = 0
        self.dead_end_local_available_step_count = 0
        self.dead_end_local_follow_step_count = 0
        self.post_flash_follow_available_step_count = 0
        self.post_flash_follow_step_count = 0
        self.post_flash_pause_step_count = 0
        self.last_debug_info = {}

        self.prev_frontier_available = False
        self.prev_frontier_action = None
        self.prev_loop_anchor_available = False
        self.prev_loop_anchor_action = None
        self.prev_loop_survival_mode = False
        self.prev_dead_end_flash_available = False
        self.prev_dead_end_flash_action = None
        self.prev_dead_end_backtrack_available = False
        self.prev_dead_end_backtrack_action = None
        self.prev_dead_end_local_available = False
        self.prev_dead_end_local_action = None
        self.prev_post_flash_follow_available = False
        self.prev_post_flash_follow_action = None
        self.last_flash_info_by_action = {}
        self.prev_flash_info_by_action = {}
        self.prev_flash_planner_reason = "NO_FLASH_MOVE_IS_BETTER"

    def get_episode_metrics(self):
        steps = max(1, self.transition_count)
        frontier_steps = max(1, self.frontier_available_step_count)
        loop_steps = max(1, self.loop_anchor_available_step_count)
        flash_steps = max(1, self.flash_action_count)
        danger_flash_steps = max(1, self.danger_flash_count)
        planner_steps = max(1, self.flash_eval_trigger_count)
        wall_cross_steps = max(1, self.wall_cross_flash_count)
        dead_end_flash_steps = max(1, self.dead_end_flash_available_step_count)
        dead_end_backtrack_steps = max(1, self.dead_end_backtrack_available_step_count)
        dead_end_local_steps = max(1, self.dead_end_local_available_step_count)
        post_flash_steps = max(1, self.post_flash_follow_available_step_count)
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
            "dead_end_backtrack_rate": round(self.dead_end_backtrack_step_count / steps, 4),
            "dead_end_local_mode_rate": round(self.dead_end_local_mode_step_count / steps, 4),
            "dead_end_local_commit_rate": round(self.dead_end_local_commit_step_count / steps, 4),
            "dead_end_flash_follow_rate": round(
                self.dead_end_flash_follow_step_count / dead_end_flash_steps, 4
            ),
            "dead_end_backtrack_follow_rate": round(
                self.dead_end_backtrack_follow_step_count / dead_end_backtrack_steps, 4
            ),
            "dead_end_local_follow_rate": round(
                self.dead_end_local_follow_step_count / dead_end_local_steps, 4
            ),
            "post_flash_follow_rate": round(
                self.post_flash_follow_step_count / post_flash_steps, 4
            ),
            "post_flash_pause_rate": round(
                self.post_flash_pause_step_count / post_flash_steps, 4
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
        }

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

        monsters = frame_state.get("monsters", [])
        monster_positions = _visible_monster_positions(monsters)
        monster_feats = []
        for idx in range(2):
            if idx < len(monsters):
                monster = monsters[idx]
                is_in_view = float(monster.get("is_in_view", 0))
                monster_pos = monster["pos"]
                dist_bucket_norm = _norm(monster.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)
                dist_norm = _safe_exact_dist(hero_pos, monster_pos) if is_in_view else dist_bucket_norm
                monster_feats.append(
                    np.array(
                        [
                            is_in_view,
                            _norm(monster_pos["x"], MAP_SIZE) if is_in_view else 0.0,
                            _norm(monster_pos["z"], MAP_SIZE) if is_in_view else 0.0,
                            _norm(monster.get("speed", 1), MAX_MONSTER_SPEED),
                            dist_norm,
                            _dir_norm(monster.get("hero_relative_direction", 0)),
                            dist_bucket_norm,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                monster_feats.append(np.zeros(7, dtype=np.float32))

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
        pressure_treasure_window_hint = bool(
            not current_true_threat
            and cur_min_dist_norm >= CRITICAL_MONSTER_DIST_NORM
            and nearest_treasure_dist_norm <= PRESSURE_TREASURE_WINDOW_DIST_NORM
            and current_local_space_score >= PRESSURE_TREASURE_MIN_SPACE_SCORE
            and (
                best_move_leaves_danger
                or best_move_safe
                or current_local_branch_factor >= LOCAL_ESCAPE_MAX_BRANCH_FACTOR
            )
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
        structural_narrow_topology_flag = bool(
            current_local_branch_factor <= LOCAL_ESCAPE_MAX_BRANCH_FACTOR
        )
        flash_constrained_flag = bool(
            blocked_flash_count >= 2
            or legal_flash_action_count <= 3
        )
        narrow_topology_flag = structural_narrow_topology_flag
        last_move_action = int(last_action % 8) if last_action >= 0 else None
        current_pos_tuple = (int(hero_pos["x"]), int(hero_pos["z"]))
        current_is_frontier = self._is_frontier(hero_pos_key)
        route_history_ready = len(self.route_history) >= LOCAL_REACTION_MIN_HISTORY
        same_cell_stall = _detect_same_cell_stall(
            self.position_history,
            current_pos_tuple,
            stall_steps=2,
        )
        two_cell_oscillation = _detect_two_cell_oscillation(
            self.position_history,
            current_pos_tuple,
        )
        tactical_reaction_ready = bool(
            route_history_ready
            or int(self.step_no) >= EARLY_TACTICAL_GRACE_STEPS
            or same_cell_stall
            or two_cell_oscillation
            or cur_min_dist_norm < CRITICAL_MONSTER_DIST_NORM
        )
        early_flash_guard = bool(
            not tactical_reaction_ready
            and cur_min_dist_norm >= CRITICAL_MONSTER_DIST_NORM
        )
        if early_flash_guard:
            for action_idx in range(8):
                legal_action[8 + action_idx] = 0
            legal_flash_action_count = 0
            flash_constrained_flag = True
        dead_end_under_pressure = bool(
            structural_narrow_topology_flag
            and current_danger
            and cur_min_dist_norm < LOOP_SURVIVAL_TRIGGER_DIST_NORM
            and tactical_reaction_ready
            and (
                current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
                or (
                    flash_constrained_flag
                    and not best_move_leaves_danger
                    and cur_min_dist_norm < FLASH_ESCAPE_TRIGGER_DIST_NORM
                )
            )
        )
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
        (
            backtrack_flag,
            backtrack_dir_norm,
            backtrack_dist_norm,
            backtrack_action,
            backtrack_quality,
            backtrack_route_steps,
            backtrack_exit_pos,
            backtrack_info,
        ) = _get_backtrack_exit_target(
            self.route_history,
            hero_pos,
            map_info,
            monster_positions,
            cur_min_dist_norm,
            current_local_branch_factor,
        )
        backtrack_available = bool(
            backtrack_action is not None
            and backtrack_flag > 0.5
            and legal_action[int(backtrack_action)] > 0
        )
        backtrack_blocked = bool(
            backtrack_available
            and (
                backtrack_info["monster_dist_norm"] < DEAD_END_BACKTRACK_MIN_DIST_NORM
                or backtrack_info["monster_gain"] < -DEAD_END_BACKTRACK_GAIN_TOLERANCE
            )
        )
        backtrack_exit_ready = bool(
            backtrack_available
            and (
                backtrack_route_steps >= BACKTRACK_EXIT_MIN_ROUTE_STEPS
                or (
                    backtrack_info["branch_factor"] > current_local_branch_factor
                    and backtrack_info["monster_gain"] >= -DEAD_END_BACKTRACK_GAIN_TOLERANCE
                )
            )
        )
        backtrack_retry_mode = bool(
            dead_end_under_pressure
            and backtrack_exit_ready
            and backtrack_blocked
            and route_history_ready
            and cur_min_dist_norm >= BACKTRACK_RETRY_SAFE_DIST_NORM
            and not same_cell_stall
            and not two_cell_oscillation
        )
        backtrack_escape_context = bool(
            backtrack_exit_ready
            and backtrack_blocked
            and not backtrack_retry_mode
        )
        flash_escape_urgent = bool(
            best_flash_action is not None
            and best_flash_info is not None
            and flash_cd <= 0
            and base_legal_action[8 + best_flash_action] > 0
            and current_danger
            and (
                current_true_threat
                or cur_min_dist_norm < CRITICAL_MONSTER_DIST_NORM
                or (
                    backtrack_escape_context
                    and not best_move_leaves_danger
                )
                or (
                    dead_end_under_pressure
                    and (
                        not best_move_leaves_danger
                        and current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
                        and cur_min_dist_norm < FLASH_ESCAPE_TRIGGER_DIST_NORM
                    )
                )
            )
        )
        flash_escape_possible = bool(
            (dead_end_under_pressure or backtrack_escape_context or cur_min_dist_norm < CRITICAL_MONSTER_DIST_NORM)
            and flash_escape_urgent
            and planner_flash_override
            and best_flash_action is not None
            and best_flash_info is not None
            and not (
                pressure_treasure_window_hint
                and best_move_leaves_danger
                and best_move_safe
            )
        )
        dead_end_backtrack_mode = bool(
            dead_end_under_pressure
            and not flash_escape_possible
            and backtrack_exit_ready
            and (not backtrack_blocked or backtrack_retry_mode)
        )
        local_escape_fallback = bool(
            backtrack_escape_context
            or (
                current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
                and route_history_ready
                and (same_cell_stall or two_cell_oscillation)
            )
        )
        dead_end_local_mode = bool(
            dead_end_under_pressure
            and not flash_escape_possible
            and not dead_end_backtrack_mode
            and local_escape_action is not None
            and local_escape_fallback
            and not pressure_treasure_window_hint
        )
        local_commit_mode = bool(
            dead_end_local_mode
            and local_escape_action is not None
            and local_escape_quality >= LOCAL_COMMIT_MIN_QUALITY
            and not current_is_frontier
            and current_local_branch_factor <= LOCAL_ESCAPE_MAX_BRANCH_FACTOR
            and (
                backtrack_escape_context
                or (
                    current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
                    and same_cell_stall
                )
            )
            and (
                cur_min_dist_norm < LOCAL_COMMIT_TRIGGER_DIST_NORM
                or same_cell_stall
                or two_cell_oscillation
            )
            and not pressure_treasure_window_hint
        )
        if local_commit_mode:
            allowed_move_actions = {int(local_escape_action), *_adjacent_actions(local_escape_action)}
            for action_idx in range(8):
                legal_action[action_idx] = 1 if (action_idx in allowed_move_actions and legal_action[action_idx]) else 0
                legal_action[8 + action_idx] = 0

        flash_commit_mode = bool(
            flash_escape_possible
            and best_flash_action is not None
            and cur_min_dist_norm < SAFE_RESOURCE_DIST_NORM
            and same_cell_stall
        )
        if flash_commit_mode:
            for action_idx in range(8):
                legal_action[action_idx] = 0
                legal_action[8 + action_idx] = 1 if action_idx == best_flash_action else 0
        elif (
            flash_escape_possible
            and best_flash_action is not None
            and best_flash_info is not None
            and not best_flash_info["invalid"]
            and (
                planner_flash_override
                or current_true_threat
                or not best_move_leaves_danger
                or same_cell_stall
                or two_cell_oscillation
            )
        ):
            soft_flash_candidates = []
            for action_idx in range(8):
                flash_info = flash_action_infos[action_idx]
                if base_legal_action[8 + action_idx] <= 0:
                    continue
                if flash_info["invalid"] or flash_info["soft_block"]:
                    continue
                if (
                    flash_info["score"] < best_flash_info["score"] - SOFT_FLASH_KEEP_SCORE_TOLERANCE
                    and not flash_info["leave_danger"]
                ):
                    continue
                soft_flash_candidates.append(
                    (
                        int(flash_info["leave_danger"]),
                        int(flash_info["leave_threat"]),
                        flash_info["score"],
                        flash_info["distance_gain"],
                        flash_info["openness_gain"],
                        action_idx,
                    )
                )
            soft_flash_candidates.sort(reverse=True)
            allowed_flash_actions = {int(best_flash_action)}
            for candidate in soft_flash_candidates[:SOFT_FLASH_KEEP_TOPK]:
                allowed_flash_actions.add(int(candidate[-1]))
            for action_idx in range(8):
                legal_action[8 + action_idx] = (
                    1 if action_idx in allowed_flash_actions else 0
                )
        else:
            # Leave the hard safety gate unchanged in V1. If soft flash
            # candidates are enabled later, this is the place to keep top-K
            # planner-approved flash actions instead of clearing the whole
            # flash branch.
            for action_idx in range(8):
                legal_action[8 + action_idx] = 0
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
        if dead_end_backtrack_mode and backtrack_action is not None:
            guidance_flag = backtrack_flag
            guidance_dir_norm = backtrack_dir_norm
            guidance_dist_norm = backtrack_dist_norm
            guidance_source = "dead_end_backtrack"
        elif local_commit_mode and local_escape_action is not None:
            guidance_flag = local_escape_flag
            guidance_dir_norm = local_escape_dir_norm
            guidance_dist_norm = local_escape_dist_norm
            guidance_source = "dead_end_local_commit"
        elif flash_commit_mode and best_flash_action is not None:
            guidance_flag = 1.0
            guidance_dir_norm = _action_to_dir_norm(best_flash_action)
            guidance_dist_norm = 1.0 - float(best_flash_info.get("landing_ratio", 0.0))
            guidance_source = "flash_planner"
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
                    and int(self.step_no) < 80
                ):
                    legal_action[8 + action_idx] = 0
                    blocked_flash_count += 1
            legal_flash_action_count = int(sum(legal_action[8:]))

        anti_oscillation_mode = bool(
            guidance_source == "frontier"
            and local_escape_action is not None
            and local_escape_quality >= ANTI_OSCILLATION_MIN_LOCAL_QUALITY
            and two_cell_oscillation
            and not dead_end_under_pressure
            and cur_min_dist_norm < LOOP_SURVIVAL_TRIGGER_DIST_NORM
            and nearest_treasure_dist_norm > ANTI_OSCILLATION_GOAL_NEAR_DIST_NORM
            and nearest_buff_dist_norm > 0.75 * ANTI_OSCILLATION_GOAL_NEAR_DIST_NORM
            and hidden_treasure_flag < 0.5
            and not (
                memory_treasure_flag > 0.5
                and memory_treasure_dist_norm <= ANTI_OSCILLATION_GOAL_NEAR_DIST_NORM
            )
        )
        if anti_oscillation_mode:
            guidance_flag = local_escape_flag
            guidance_dir_norm = local_escape_dir_norm
            guidance_dist_norm = local_escape_dist_norm
            guidance_source = "anti_oscillation"

        frontier_local_commit_mode = bool(
            guidance_source in {"frontier", "anti_oscillation"}
            and dead_end_under_pressure
            and not flash_escape_possible
            and not dead_end_backtrack_mode
            and local_escape_action is not None
            and local_escape_quality >= LOCAL_COMMIT_MIN_QUALITY
            and not current_is_frontier
            and current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
            and route_history_ready
            and backtrack_escape_context
            and cur_min_dist_norm < LOCAL_COMMIT_TRIGGER_DIST_NORM
            and (
                same_cell_stall
                or two_cell_oscillation
            )
        )
        if frontier_local_commit_mode and not local_commit_mode:
            guidance_flag = local_escape_flag
            guidance_dir_norm = local_escape_dir_norm
            guidance_dist_norm = local_escape_dist_norm
            guidance_source = "dead_end_local"

        move_dist_norm = _safe_exact_dist(hero_pos, self.last_hero_pos) if self.last_hero_pos else 0.0
        prev_flash_info = (
            self.prev_flash_info_by_action.get(last_action % 8, {})
            if last_action >= 8
            else {}
        )
        flash_escape_gain = 0.0
        flash_margin_gain = 0.0
        flash_openness_gain = 0.0
        flash_wasted = bool(
            last_action >= 8
            and (
                move_dist_norm < FLASH_WASTED_MOVE_NORM
                or int(prev_flash_info.get("landing_step", 0)) <= 0
            )
        )
        flash_effective = False
        flash_danger = False

        post_flash_follow_action = None
        if last_action >= 8 and not flash_wasted:
            if dead_end_backtrack_mode and backtrack_action is not None:
                post_flash_follow_action = int(backtrack_action)
            elif local_commit_mode and local_escape_action is not None:
                post_flash_follow_action = int(local_escape_action)
            elif dead_end_local_mode and local_escape_action is not None:
                post_flash_follow_action = int(local_escape_action)
            elif anti_oscillation_mode and local_escape_action is not None:
                post_flash_follow_action = int(local_escape_action)
            elif loop_survival_mode and loop_action is not None:
                post_flash_follow_action = int(loop_action)
            elif frontier_action is not None:
                post_flash_follow_action = int(frontier_action)
            elif last_move_action is not None and legal_action[last_move_action]:
                post_flash_follow_action = int(last_move_action)

        post_flash_follow_mode = bool(
            post_flash_follow_action is not None
            and legal_action[int(post_flash_follow_action)] > 0
        )
        if post_flash_follow_mode:
            guidance_flag = 1.0
            guidance_dir_norm = _action_to_dir_norm(post_flash_follow_action)
            guidance_dist_norm = 0.0
            guidance_source = "post_flash_follow"

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
            ],
            dtype=np.float32,
        )

        last_action_feat = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        if 0 <= int(last_action) < Config.ACTION_NUM:
            last_action_feat[int(last_action)] = 1.0

        temporal_feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                risk_summary_feat,
                last_action_feat,
            ]
        ).astype(np.float32)

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

        if last_action >= 8:
            flash_escape_gain = cur_min_dist_norm - self.last_min_monster_dist_norm
            flash_margin_gain = current_margin_cells - self.last_min_margin_cells
            flash_openness_gain = current_local_space_score - self.last_local_space_score
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
            if dead_end_under_pressure:
                if flash_escape_possible:
                    self.dead_end_flash_escape_step_count += 1
                elif dead_end_backtrack_mode:
                    self.dead_end_backtrack_step_count += 1
                elif dead_end_local_mode:
                    self.dead_end_local_mode_step_count += 1
            if local_commit_mode:
                self.dead_end_local_commit_step_count += 1
            if self.prev_dead_end_flash_available and self.prev_dead_end_flash_action is not None:
                self.dead_end_flash_available_step_count += 1
                if last_action >= 8 and (last_action % 8) == self.prev_dead_end_flash_action:
                    self.dead_end_flash_follow_step_count += 1
            if self.prev_dead_end_backtrack_available and self.prev_dead_end_backtrack_action is not None:
                self.dead_end_backtrack_available_step_count += 1
                if last_action >= 0 and (last_action % 8) == self.prev_dead_end_backtrack_action:
                    self.dead_end_backtrack_follow_step_count += 1
            if self.prev_dead_end_local_available and self.prev_dead_end_local_action is not None:
                self.dead_end_local_available_step_count += 1
                if last_action >= 0 and (last_action % 8) == self.prev_dead_end_local_action:
                    self.dead_end_local_follow_step_count += 1
            if self.prev_post_flash_follow_available and self.prev_post_flash_follow_action is not None:
                self.post_flash_follow_available_step_count += 1
                post_flash_follow_hit = bool(
                    last_action >= 0
                    and last_action < 8
                    and (last_action % 8) == self.prev_post_flash_follow_action
                    and move_dist_norm >= POST_FLASH_FOLLOW_MIN_DIST_NORM
                )
                if post_flash_follow_hit:
                    self.post_flash_follow_step_count += 1
                elif move_dist_norm < POST_FLASH_PAUSE_DIST_NORM:
                    self.post_flash_pause_step_count += 1
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
                else:
                    self.close_escape_flash_count += 1
                if (
                    current_local_branch_factor <= LOCAL_DEAD_END_MAX_BRANCH_FACTOR
                    and current_local_space_score <= DEAD_END_SPACE_SCORE
                ):
                    self.post_flash_dead_end_count += 1

        survive_reward = HIGH_PRESSURE_SURVIVE_REWARD
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        treasure_score = float(hero.get("treasure_score", env_info.get("treasure_score", 0.0)))
        treasure_reward = TREASURE_SCORE_REWARD_SCALE * max(
            0.0, treasure_score - self.last_treasure_score
        )

        pressure_treasure_window = bool(
            dead_end_under_pressure and pressure_treasure_window_hint
        )
        safe_for_treasure = bool(
            cur_min_dist_norm >= SAFE_RESOURCE_DIST_NORM
            or pressure_treasure_window
        )
        buff_focus = buff_remain <= 0 or cur_min_dist_norm < SAFE_RESOURCE_DIST_NORM

        treasure_approach_reward = TREASURE_APPROACH_REWARD_SCALE * max(
            0.0, self.last_nearest_treasure_dist_norm - nearest_treasure_dist_norm
        )
        if not safe_for_treasure:
            treasure_approach_reward *= UNSAFE_TREASURE_APPROACH_SCALE

        buff_approach_reward = BUFF_APPROACH_REWARD_SCALE * max(
            0.0, self.last_nearest_buff_dist_norm - nearest_buff_dist_norm
        )
        if not buff_focus:
            buff_approach_reward *= PASSIVE_BUFF_APPROACH_SCALE

        path_follow_reward = 0.0
        if last_action >= 0:
            action_dir = last_action % 8
            if safe_for_treasure:
                path_follow_reward += TREASURE_PATH_FOLLOW_REWARD * float(
                    treasure_path_scores[action_dir]
                )
            if buff_focus:
                path_follow_reward += BUFF_PATH_FOLLOW_REWARD * float(
                    buff_path_scores[action_dir]
                )

        buff_reward = 0.25 if buff_remain > self.last_buff_remain + 1e-6 else 0.0
        if visit_count == 1:
            explore_reward = EXPLORE_REWARD_ON_NEW_CELL
        else:
            revisit_penalty = -REVISIT_PENALTY_SCALE * min(5, visit_count - 1)
            if loop_survival_mode and loop_flag > 0.5:
                revisit_penalty *= LOOP_REVISIT_PENALTY_DISCOUNT
            if dead_end_backtrack_mode and backtrack_action is not None:
                revisit_penalty *= BACKTRACK_REVISIT_PENALTY_DISCOUNT
            if dead_end_local_mode and local_escape_action is not None:
                revisit_penalty *= DEAD_END_LOCAL_REVISIT_PENALTY_DISCOUNT
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

        dead_end_backtrack_reward = 0.0
        if (
            self.prev_dead_end_backtrack_available
            and self.prev_dead_end_backtrack_action is not None
            and last_action >= 0
        ):
            action_dir = last_action % 8
            if action_dir == self.prev_dead_end_backtrack_action:
                dead_end_backtrack_reward += 0.03
            elif move_dist_norm < 0.01:
                dead_end_backtrack_reward -= 0.05
            elif action_dir == _opposite_action(self.prev_dead_end_backtrack_action):
                dead_end_backtrack_reward -= 0.03

        loop_follow_reward = 0.0
        if last_action >= 0 and loop_survival_mode and loop_action is not None:
            action_dir = last_action % 8
            if action_dir == loop_action:
                loop_follow_reward += 0.06 * (0.6 + 0.4 * max(0.0, 1.0 - loop_dist_norm))
            elif move_dist_norm < 0.01:
                loop_follow_reward -= 0.08
            elif action_dir == _opposite_action(loop_action):
                loop_follow_reward -= 0.03

        post_flash_follow_reward = 0.0
        if (
            self.prev_post_flash_follow_available
            and self.prev_post_flash_follow_action is not None
            and last_action >= 0
        ):
            action_dir = last_action % 8
            if (
                last_action < 8
                and action_dir == self.prev_post_flash_follow_action
                and move_dist_norm >= POST_FLASH_FOLLOW_MIN_DIST_NORM
            ):
                post_flash_follow_reward += 0.03
            elif move_dist_norm < POST_FLASH_PAUSE_DIST_NORM:
                post_flash_follow_reward -= 0.04

        dead_end_local_reward = 0.0
        if last_action >= 0 and dead_end_local_mode and local_escape_action is not None:
            action_dir = last_action % 8
            adjacent_local_actions = set(_adjacent_actions(local_escape_action))
            if action_dir == local_escape_action:
                scale = 0.10 if local_commit_mode else 0.07
                dead_end_local_reward += scale * (0.5 + 0.5 * local_escape_quality)
            elif local_commit_mode and action_dir in adjacent_local_actions:
                dead_end_local_reward += 0.04 * (0.5 + 0.5 * local_escape_quality)
            elif move_dist_norm < 0.01:
                dead_end_local_reward -= 0.12 if local_commit_mode else 0.10
            elif action_dir == _opposite_action(local_escape_action):
                dead_end_local_reward -= 0.06 if local_commit_mode else 0.04

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
        self.last_danger_state = current_danger
        self.last_true_threat_state = current_true_threat
        self.last_near_threat_state = current_near_threat
        self.last_treasure_score = treasure_score
        self.last_buff_remain = float(buff_remain)
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
        self.position_history.append(current_pos_tuple)
        if not self.route_history or self.route_history[-1] != current_pos_tuple:
            self.route_history.append(current_pos_tuple)
        self.last_nearest_treasure_dist_norm = nearest_treasure_dist_norm
        self.last_nearest_buff_dist_norm = nearest_buff_dist_norm
        self.prev_frontier_available = frontier_action is not None
        self.prev_frontier_action = frontier_action
        self.prev_loop_anchor_available = loop_action is not None
        self.prev_loop_anchor_action = loop_action
        self.prev_loop_survival_mode = loop_survival_mode
        self.prev_dead_end_flash_available = flash_escape_possible and best_flash_action is not None
        self.prev_dead_end_flash_action = best_flash_action
        self.prev_dead_end_backtrack_available = dead_end_backtrack_mode and backtrack_action is not None
        self.prev_dead_end_backtrack_action = backtrack_action
        self.prev_dead_end_local_available = dead_end_local_mode and local_escape_action is not None
        self.prev_dead_end_local_action = local_escape_action
        self.prev_post_flash_follow_available = (
            post_flash_follow_mode and post_flash_follow_action is not None
        )
        self.prev_post_flash_follow_action = post_flash_follow_action
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
            "current_true_threat": int(current_true_threat),
            "current_near_threat": int(current_near_threat),
            "current_danger": int(current_danger),
            "current_margin_cells": round(float(current_margin_cells), 4),
            "current_local_space_score": round(float(current_local_space_score), 4),
            "local_dead_end": int(local_dead_end),
            "narrow_topology_flag": int(narrow_topology_flag),
            "dead_end_under_pressure": int(dead_end_under_pressure),
            "flash_escape_urgent": int(flash_escape_urgent),
            "flash_eval_trigger": int(flash_eval_trigger),
            "flash_planner_override": int(planner_flash_override),
            "flash_commit_mode": int(flash_commit_mode),
            "post_flash_follow_mode": int(post_flash_follow_mode),
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
            "dead_end_backtrack_mode": int(dead_end_backtrack_mode),
            "dead_end_backtrack_action": int(backtrack_action) if backtrack_action is not None else -1,
            "dead_end_backtrack_ready": int(backtrack_exit_ready),
            "dead_end_backtrack_dist_norm": round(float(backtrack_dist_norm), 4),
            "dead_end_backtrack_quality": round(float(backtrack_quality), 4),
            "dead_end_backtrack_blocked": int(backtrack_blocked),
            "dead_end_local_mode": int(dead_end_local_mode),
            "dead_end_local_commit_mode": int(local_commit_mode),
            "dead_end_flash_action": int(best_flash_action) if flash_escape_possible and best_flash_action is not None else -1,
            "local_escape_action": int(local_escape_action) if local_escape_action is not None else -1,
            "local_escape_quality": round(float(local_escape_quality), 4),
            "local_escape_dist_norm": round(float(local_escape_dist_norm), 4),
            "post_flash_follow_action": int(post_flash_follow_action)
            if post_flash_follow_action is not None
            else -1,
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
            + dead_end_backtrack_reward
            + post_flash_follow_reward
            + loop_follow_reward
            + dead_end_local_reward
            + anti_oscillation_reward
        ]

        return feature, temporal_feature, legal_action, reward
