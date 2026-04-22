#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Flash escape strategy V1 for the DIY PPO agent.

This module is intentionally framework-light. It only consumes semantic
state/context and returns planner diagnostics plus a policy bias vector.
"""

import numpy as np

from agent_diy.conf.conf import Config


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
FLASH_DISTANCES = [10, 8, 10, 8, 10, 8, 10, 8]

INF_DISTANCE = 1e6
MAX_OPENNESS_DEPTH = 3


def _distance(pos_a, pos_b):
    return float(np.sqrt((pos_a["x"] - pos_b["x"]) ** 2 + (pos_a["z"] - pos_b["z"]) ** 2))


def _metric_is_valid(value):
    return bool(np.isfinite(value) and abs(float(value)) < INF_DISTANCE * 0.5)


def _safe_gain(prev_value, next_value):
    if not (_metric_is_valid(prev_value) and _metric_is_valid(next_value)):
        return 0.0, False
    return float(next_value - prev_value), True


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


class FlashEscapeStrategy:
    def build_context(self, observation, legal_action):
        frame_state = observation["frame_state"]
        map_info = observation.get("map_info")
        hero = frame_state["heroes"]
        hero_pos = {
            "x": float(hero["pos"]["x"]),
            "z": float(hero["pos"]["z"]),
        }
        center = len(map_info) // 2 if map_info is not None and len(map_info) > 0 else 0
        monsters = self._extract_visible_monsters(frame_state.get("monsters", []))
        threat_metrics = self._threat_metrics(hero_pos, monsters)
        topology = self._topology(map_info, (center, center))

        return {
            "hero_pos": hero_pos,
            "map_info": map_info,
            "center": center,
            "legal_action": [int(v) for v in legal_action],
            "step_no": int(observation.get("step_no", 0)),
            "flash_ready": any(int(v) for v in legal_action[8:16]),
            "flash_cooldown": float(hero.get("flash_cooldown", 0.0)),
            "move_speed": 2 if hero.get("buff_remaining_time", 0) > 0 else 1,
            "monsters": monsters,
            "visible_monster_count": len(monsters),
            "threat_metrics_valid": bool(monsters),
            "min_euclid": threat_metrics["min_euclid"],
            "min_cheb": threat_metrics["min_cheb"],
            "min_margin": threat_metrics["min_margin"],
            "in_threat": threat_metrics["in_threat"],
            "in_near_threat": threat_metrics["in_near_threat"],
            "reachable_area": topology["reachable_area"],
            "exit_count": topology["exit_count"],
            "is_dead_end": topology["is_dead_end"],
            "is_choke": topology["is_choke"],
        }

    def evaluate(self, context):
        move_candidates = [self._score_candidate(context, action) for action in range(8)]
        flash_candidates = [self._score_candidate(context, action) for action in range(8, 16)]
        candidate_lookup = {
            candidate["action"]: candidate
            for candidate in move_candidates + flash_candidates
        }

        best_move = self._best_candidate(move_candidates)
        valid_flash_candidates = [candidate for candidate in flash_candidates if candidate["legal"] and not candidate["invalid"]]
        best_flash = self._best_candidate(valid_flash_candidates or flash_candidates)
        flash_eval_triggered, trigger_reason = self._should_trigger_flash(context, best_move, best_flash)

        result = {
            "flash_eval_triggered": bool(flash_eval_triggered),
            "flash_execute": False,
            "chosen_action": int(best_move["action"]) if best_move else 0,
            "chosen_action_type": "move",
            "best_move_action": int(best_move["action"]) if best_move else -1,
            "best_move_score": float(best_move["score"]) if best_move else -INF_DISTANCE,
            "best_flash_action": int(best_flash["action"]) if best_flash else -1,
            "best_flash_score": float(best_flash["score"]) if best_flash else -INF_DISTANCE,
            "flash_skip_reason": "",
            "wall_cross": False,
            "choke_escape": False,
            "leave_threat": False,
            "distance_gain": 0.0,
            "min_margin_gain": 0.0,
            "openness_gain": 0.0,
            "post_flash_dead_end": False,
            "early_flash": False,
            "decision_tag": "MOVE_BETTER",
            "trigger_reason": trigger_reason,
            "best_flash_better_than_move": False,
            "best_move_better_than_flash": True,
            "action_bias": [0.0] * Config.ACTION_NUM,
            "_candidate_lookup": candidate_lookup,
        }

        if not flash_eval_triggered:
            result["flash_skip_reason"] = "NOT_TRIGGERED"
            result["decision_tag"] = "MOVE_BETTER"
            result["action_bias"] = self._build_action_bias(context, best_move, best_flash, result)
            return result

        if not context["flash_ready"]:
            result["flash_skip_reason"] = "COOLDOWN"
            result["decision_tag"] = "MOVE_BETTER"
            result["action_bias"] = self._build_action_bias(context, best_move, best_flash, result)
            return result

        if best_flash is None or best_flash["invalid"] or not best_flash["legal"]:
            result["flash_skip_reason"] = "NO_SAFE_CANDIDATE"
            result["decision_tag"] = "NO_SAFE_FLASH"
            result["action_bias"] = self._build_action_bias(context, best_move, best_flash, result)
            return result

        best_flash_better_than_move = best_flash["score"] >= best_move["score"] + Config.FLASH_OVERRIDE_MARGIN
        urgent_escape = context["in_threat"] and best_flash["leave_threat"] and not best_move["leave_threat"]
        flash_has_escape_value = self._has_escape_value(best_flash, context)

        result["best_flash_better_than_move"] = bool(best_flash_better_than_move or urgent_escape)
        result["best_move_better_than_flash"] = not result["best_flash_better_than_move"]

        if result["best_flash_better_than_move"] and flash_has_escape_value:
            result["flash_execute"] = True
            result["chosen_action"] = int(best_flash["action"])
            result["chosen_action_type"] = "flash"
            result["wall_cross"] = bool(best_flash["wall_cross"])
            result["choke_escape"] = bool(best_flash["choke_escape"])
            result["leave_threat"] = bool(best_flash["leave_threat"])
            result["distance_gain"] = float(best_flash["distance_gain"])
            result["min_margin_gain"] = float(best_flash["min_margin_gain"])
            result["openness_gain"] = float(best_flash["openness_gain"])
            result["post_flash_dead_end"] = bool(best_flash["is_dead_end"])
            result["early_flash"] = bool(context["step_no"] <= Config.FLASH_EARLY_STEP_THRESHOLD)
            if best_flash["wall_cross"]:
                result["decision_tag"] = "WALL_CROSS_ESCAPE"
            elif best_flash["choke_escape"]:
                result["decision_tag"] = "CHOKE_ESCAPE"
            else:
                result["decision_tag"] = "CLOSE_ESCAPE"
        else:
            result["flash_skip_reason"] = "MOVE_BETTER"
            result["decision_tag"] = "MOVE_BETTER"

        result["action_bias"] = self._build_action_bias(context, best_move, best_flash, result)
        return result

    def evaluate_transition(self, prev_context, decision_record, action, next_context):
        info = self._empty_transition_info()
        planner_result = (decision_record or {}).get("planner_result", {})
        decision_info = (decision_record or {}).get("decision_info", {})
        candidate_lookup = planner_result.get("_candidate_lookup", {})
        candidate = candidate_lookup.get(int(action))
        flash_used = int(action) >= 8

        info.update(decision_info)
        info["flash_used"] = bool(flash_used)
        info["flash"] = int(flash_used)
        info["selected_action"] = int(action)
        info["selected_action_type"] = "flash" if flash_used else "move"
        visible_monsters = int(prev_context.get("visible_monster_count", 0))
        info["unknown_flash"] = int(flash_used and visible_monsters == 0)
        info["flash_in_danger"] = bool(
            flash_used
            and visible_monsters > 0
            and (
                prev_context["in_threat"]
                or prev_context["in_near_threat"]
                or prev_context["min_euclid"] <= Config.FLASH_TRIGGER_DANGER_DISTANCE
            )
        )
        info["flash_in_safe"] = bool(
            flash_used and visible_monsters > 0 and not info["flash_in_danger"]
        )
        info["safe_flash"] = int(info["flash_in_safe"])
        info["danger_flash"] = int(info["flash_in_danger"])

        if not flash_used:
            return info

        distance_gain, distance_gain_valid = _safe_gain(
            prev_context["min_euclid"], next_context["min_euclid"]
        )
        min_margin_gain, min_margin_gain_valid = _safe_gain(
            prev_context["min_margin"], next_context["min_margin"]
        )
        openness_gain = next_context["reachable_area"] - prev_context["reachable_area"]
        pre_in_threat = bool(prev_context["in_threat"])
        # `pre_in_near_threat` deliberately reuses the planner's current
        # near-danger semantics from `_threat_metrics`: it is the ring outside
        # the monster 3x3 threat zone but still inside the planner's
        # near-threat band.
        pre_in_near_threat = bool(not pre_in_threat and prev_context["in_near_threat"])
        leave_threat = bool(pre_in_threat and not next_context["in_threat"])
        leave_danger = bool(
            (pre_in_threat or pre_in_near_threat)
            and not next_context["in_threat"]
            and not next_context["in_near_threat"]
        )
        leave_near_threat = bool(
            prev_context["in_near_threat"]
            and not prev_context["in_threat"]
            and not next_context["in_threat"]
            and not next_context["in_near_threat"]
        )
        post_flash_dead_end = bool(next_context["is_dead_end"])
        invalid_flash = False
        gain_unavailable_reason = ""

        if not distance_gain_valid and not min_margin_gain_valid:
            if prev_context.get("visible_monster_count", 0) <= 0 and next_context.get("visible_monster_count", 0) <= 0:
                gain_unavailable_reason = "NO_VISIBLE_MONSTER_PRE_POST"
            elif prev_context.get("visible_monster_count", 0) <= 0:
                gain_unavailable_reason = "NO_VISIBLE_MONSTER_PRE"
            elif next_context.get("visible_monster_count", 0) <= 0:
                gain_unavailable_reason = "NO_VISIBLE_MONSTER_POST"
            else:
                gain_unavailable_reason = "INVALID_THREAT_METRIC"

        if candidate is None:
            invalid_flash = True
        else:
            if candidate["move_distance"] < 1.5:
                invalid_flash = True
            if next_context["in_threat"] and not leave_threat:
                invalid_flash = True
            if distance_gain_valid and distance_gain < 0.0 and not candidate["wall_cross"]:
                invalid_flash = True
            if post_flash_dead_end and openness_gain <= 0.0:
                invalid_flash = True

        # `flash_leave_threat` is the strictest post-step escape signal: it only
        # counts cases where the agent was in the monster 3x3 threat zone before
        # the flash and the next observation is no longer in that threat zone.
        # `escape_effective` is broader and allows strong danger improvements
        # without a full threat exit. `danger_effective` is broader still, but
        # it must still reflect concrete safety improvement rather than a raw
        # planner score.
        escape_effective_flash = bool(
            info["flash_in_danger"]
            and not invalid_flash
            and (
                leave_threat
                or (
                    leave_near_threat
                    and min_margin_gain_valid
                    and min_margin_gain >= Config.FLASH_ESCAPE_MIN_MARGIN_GAIN
                )
                or (
                    distance_gain_valid
                    and min_margin_gain_valid
                    and distance_gain >= Config.FLASH_ESCAPE_DISTANCE_GAIN
                    and min_margin_gain >= Config.FLASH_ESCAPE_MIN_MARGIN_GAIN
                    and candidate is not None
                    and (candidate["wall_cross"] or candidate["choke_escape"])
                )
            )
        )
        danger_effective_flash = bool(
            info["flash_in_danger"]
            and not invalid_flash
            and (
                leave_threat
                or escape_effective_flash
                or (
                    not next_context["in_threat"]
                    and min_margin_gain_valid
                    and min_margin_gain >= Config.FLASH_DANGER_EFFECTIVE_MIN_MARGIN_GAIN
                    and (
                        (distance_gain_valid and distance_gain >= Config.FLASH_DANGER_EFFECTIVE_DISTANCE_GAIN)
                        or (candidate is not None and (candidate["wall_cross"] or candidate["choke_escape"]))
                    )
                )
            )
        )
        flash_effective = bool(
            not invalid_flash
            and (
                danger_effective_flash
                or (
                    not info["flash_in_danger"]
                    and candidate is not None
                    and (candidate["wall_cross"] or candidate["choke_escape"])
                    and distance_gain_valid
                    and min_margin_gain_valid
                    and distance_gain >= Config.FLASH_ESCAPE_DISTANCE_GAIN
                    and min_margin_gain >= Config.FLASH_ESCAPE_MIN_MARGIN_GAIN
                )
            )
        )
        danger_ineffective_flash = bool(info["flash_in_danger"] and not flash_effective)
        flash_reward = 0.0
        if not Config.DISABLE_LEGACY_FLASH_REWARD:
            flash_reward = self._legacy_flash_reward(
                flash_effective=flash_effective,
                leave_threat=leave_threat,
                invalid_flash=invalid_flash,
            )

        info.update(
            {
                "flash_effective": bool(flash_effective),
                "effective_flash": int(flash_effective),
                "ineffective_flash": int(not flash_effective),
                "danger_effective_flash": bool(danger_effective_flash),
                "danger_ineffective_flash": bool(danger_ineffective_flash),
                "flash_effective_public": int(flash_effective),
                "flash_distance_delta": float(distance_gain),
                "flash_distance_delta_raw": float(distance_gain),
                "distance_gain_valid": bool(distance_gain_valid),
                "distance_gain": float(distance_gain),
                "min_margin_gain_valid": bool(min_margin_gain_valid),
                "min_margin_gain": float(min_margin_gain),
                "gain_unavailable_reason": gain_unavailable_reason,
                "openness_gain": float(openness_gain),
                "flash_min_margin_gain": float(min_margin_gain),
                "flash_openness_gain": float(openness_gain),
                "flash_pre_in_threat": int(pre_in_threat),
                "flash_pre_in_near_threat": int(pre_in_near_threat),
                "leave_threat": bool(leave_threat),
                "flash_leave_threat": int(leave_threat),
                "leave_danger": bool(leave_danger),
                "flash_leave_danger": int(leave_danger),
                "leave_near_threat": bool(leave_near_threat),
                "post_flash_dead_end": bool(post_flash_dead_end),
                "wall_cross": bool(candidate["wall_cross"] if candidate else False),
                "choke_escape": bool(candidate["choke_escape"] if candidate else False),
                "wall_cross_flash": int(candidate["wall_cross"] if candidate else False),
                "choke_escape_flash": int(candidate["choke_escape"] if candidate else False),
                "wall_cross_effective": int(bool(candidate and candidate["wall_cross"] and flash_effective)),
                "choke_escape_effective": int(bool(candidate and candidate["choke_escape"] and flash_effective)),
                "early_flash": int(prev_context["step_no"] <= Config.FLASH_EARLY_STEP_THRESHOLD),
                "invalid_flash": int(invalid_flash),
                "invalid_flash_penalty": int(invalid_flash),
                "escape_effective_flash": int(escape_effective_flash),
                "escape_flash": int(escape_effective_flash),
                "flash_reward": float(flash_reward),
            }
        )
        return info

    def _extract_visible_monsters(self, monsters):
        visible_monsters = []
        for monster in monsters:
            if not monster.get("is_in_view", 0):
                continue
            pos = monster.get("pos")
            if not isinstance(pos, dict) or "x" not in pos or "z" not in pos:
                continue
            visible_monsters.append(
                {
                    "pos": {
                        "x": float(pos["x"]),
                        "z": float(pos["z"]),
                    },
                    "speed": float(monster.get("speed", 1.0)),
                }
            )
        return visible_monsters

    def _score_candidate(self, context, action):
        legal = bool(context["legal_action"][action])
        is_flash = action >= 8
        (
            landing_local,
            landing_abs,
            move_distance,
            blocked_on_path,
            crossed_blocked_segment,
        ) = self._simulate_action(context, action)
        threat = self._threat_metrics(landing_abs, context["monsters"])
        topology = self._topology(context["map_info"], landing_local)

        distance_gain, distance_gain_valid = _safe_gain(
            context["min_euclid"], threat["min_euclid"]
        )
        min_margin_gain, min_margin_gain_valid = _safe_gain(
            context["min_margin"], threat["min_margin"]
        )
        openness_gain = topology["reachable_area"] - context["reachable_area"]
        leave_threat = bool(context["in_threat"] and not threat["in_threat"])
        leave_near_threat = bool(context["in_near_threat"] and threat["min_cheb"] > 2.0)
        wall_cross = bool(
            is_flash and crossed_blocked_segment and move_distance >= 1.5
        )
        choke_escape = bool(
            context["is_choke"]
            and openness_gain >= Config.FLASH_MIN_OPENNESS_GAIN
            and topology["exit_count"] > context["exit_count"]
        )
        invalid = False
        invalid_reason = ""

        if not legal:
            invalid = True
            invalid_reason = "ILLEGAL"
        elif move_distance < (1.5 if is_flash else 0.5):
            invalid = True
            invalid_reason = "NO_DISPLACEMENT"
        elif is_flash and threat["min_cheb"] <= 1.0 and not leave_threat:
            invalid = True
            invalid_reason = "STILL_IN_THREAT"
        elif is_flash and distance_gain_valid and distance_gain < 0.0 and not wall_cross:
            invalid = True
            invalid_reason = "CLOSER_TO_MONSTER"
        elif is_flash and topology["is_dead_end"] and openness_gain < Config.FLASH_MIN_OPENNESS_GAIN:
            invalid = True
            invalid_reason = "DEAD_END"

        leave_threat_score = 0.0
        if context["in_threat"]:
            leave_threat_score = (
                Config.FLASH_LEAVE_THREAT_SCORE_WEIGHT if leave_threat else -4.0
            )
        elif context["in_near_threat"]:
            leave_threat_score = (
                0.5 * Config.FLASH_LEAVE_THREAT_SCORE_WEIGHT if leave_near_threat else 0.0
            )

        distance_gain_score = float(
            np.clip(distance_gain * Config.FLASH_DISTANCE_GAIN_WEIGHT, -4.0, 4.0)
        ) if distance_gain_valid else 0.0
        min_margin_score = float(
            np.clip(min_margin_gain * Config.FLASH_MIN_MARGIN_WEIGHT, -3.0, 3.0)
        ) if min_margin_gain_valid else 0.0
        openness_score = float(
            np.clip(openness_gain * Config.FLASH_OPENNESS_WEIGHT, -3.0, 3.0)
        )
        wall_cross_score = float(
            Config.FLASH_WALL_CROSS_BONUS if wall_cross and (leave_threat or min_margin_gain >= 0.0) else 0.0
        )
        choke_escape_score = float(
            Config.FLASH_CHOKE_ESCAPE_BONUS if choke_escape else 0.0
        )
        dead_end_penalty = 3.0 if topology["is_dead_end"] else 0.0
        invalid_flash_penalty = Config.FLASH_INVALID_PENALTY_WEIGHT if invalid else 0.0

        score = (
            leave_threat_score
            + distance_gain_score
            + min_margin_score
            + openness_score
            + wall_cross_score
            + choke_escape_score
            - dead_end_penalty
            - invalid_flash_penalty
        )

        return {
            "action": int(action),
            "action_type": "flash" if is_flash else "move",
            "legal": legal,
            "score": float(score),
            "landing_local": landing_local,
            "landing_pos": landing_abs,
            "move_distance": float(move_distance),
            "blocked_on_path": bool(blocked_on_path),
            "crossed_blocked_segment": bool(crossed_blocked_segment),
            "min_euclid": float(threat["min_euclid"]),
            "min_cheb": float(threat["min_cheb"]),
            "min_margin": float(threat["min_margin"]),
            "distance_gain": float(distance_gain),
            "distance_gain_valid": bool(distance_gain_valid),
            "min_margin_gain": float(min_margin_gain),
            "min_margin_gain_valid": bool(min_margin_gain_valid),
            "openness_gain": float(openness_gain),
            "reachable_area": int(topology["reachable_area"]),
            "exit_count": int(topology["exit_count"]),
            "is_dead_end": bool(topology["is_dead_end"]),
            "leave_threat": bool(leave_threat),
            "leave_near_threat": bool(leave_near_threat),
            "wall_cross": bool(wall_cross),
            "choke_escape": bool(choke_escape),
            "invalid": bool(invalid),
            "invalid_reason": invalid_reason,
        }

    def _simulate_action(self, context, action):
        map_info = context["map_info"]
        center = context["center"]
        row, col = center, center
        dx, dz = MOVE_DELTAS[action % 8]
        blocked_on_path = False
        first_blocked_step = None
        landing_step = 0

        if action >= 8:
            max_distance = FLASH_DISTANCES[action % 8]
            landing_row, landing_col = row, col
            for step in range(1, max_distance + 1):
                probe_row = row + dz * step
                probe_col = col + dx * step
                if not _is_passable(map_info, probe_row, probe_col):
                    blocked_on_path = True
                    if first_blocked_step is None:
                        first_blocked_step = step
            for step in range(max_distance, 0, -1):
                probe_row = row + dz * step
                probe_col = col + dx * step
                if _is_passable(map_info, probe_row, probe_col):
                    landing_row, landing_col = probe_row, probe_col
                    landing_step = step
                    break
            row, col = landing_row, landing_col
        else:
            move_speed = int(max(1, context["move_speed"]))
            for _ in range(move_speed):
                next_row = row + dz
                next_col = col + dx
                if not _can_step(map_info, row, col, next_row, next_col, dx, dz):
                    break
                row, col = next_row, next_col

        landing_abs = {
            "x": context["hero_pos"]["x"] + (col - center),
            "z": context["hero_pos"]["z"] + (row - center),
        }
        move_distance = _distance(context["hero_pos"], landing_abs)
        crossed_blocked_segment = bool(
            action >= 8
            and first_blocked_step is not None
            and landing_step > first_blocked_step
        )
        return (row, col), landing_abs, move_distance, blocked_on_path, crossed_blocked_segment

    def _topology(self, map_info, local_pos):
        row, col = local_pos
        exit_count = 0
        for dx, dz in MOVE_DELTAS:
            if _can_step(map_info, row, col, row + dz, col + dx, dx, dz):
                exit_count += 1

        if map_info is None or len(map_info) == 0 or not _is_passable(map_info, row, col):
            return {
                "reachable_area": 0,
                "exit_count": 0,
                "is_dead_end": True,
                "is_choke": True,
            }

        queue = [(row, col, 0)]
        visited = {(row, col)}
        while queue:
            cur_row, cur_col, depth = queue.pop(0)
            if depth >= MAX_OPENNESS_DEPTH:
                continue
            for dx, dz in MOVE_DELTAS:
                next_row = cur_row + dz
                next_col = cur_col + dx
                if (next_row, next_col) in visited:
                    continue
                if not _can_step(map_info, cur_row, cur_col, next_row, next_col, dx, dz):
                    continue
                visited.add((next_row, next_col))
                queue.append((next_row, next_col, depth + 1))

        reachable_area = len(visited)
        return {
            "reachable_area": reachable_area,
            "exit_count": exit_count,
            "is_dead_end": reachable_area <= Config.FLASH_DEAD_END_THRESHOLD or exit_count <= 1,
            "is_choke": reachable_area <= Config.FLASH_DEAD_END_THRESHOLD + 2 or exit_count <= 2,
        }

    def _threat_metrics(self, hero_pos, monsters):
        if not monsters:
            return {
                "min_euclid": INF_DISTANCE,
                "min_cheb": INF_DISTANCE,
                "min_margin": INF_DISTANCE,
                "in_threat": False,
                "in_near_threat": False,
            }

        min_euclid = INF_DISTANCE
        min_cheb = INF_DISTANCE
        for monster in monsters:
            dx = abs(hero_pos["x"] - monster["pos"]["x"])
            dz = abs(hero_pos["z"] - monster["pos"]["z"])
            min_euclid = min(min_euclid, float(np.sqrt(dx * dx + dz * dz)))
            min_cheb = min(min_cheb, float(max(dx, dz)))

        min_margin = min_cheb - 1.0
        return {
            "min_euclid": min_euclid,
            "min_cheb": min_cheb,
            "min_margin": min_margin,
            "in_threat": bool(min_cheb <= Config.FLASH_THREAT_CHEB_DISTANCE),
            "in_near_threat": bool(
                Config.FLASH_THREAT_CHEB_DISTANCE
                < min_cheb
                <= Config.FLASH_NEAR_THREAT_CHEB_DISTANCE
            ),
        }

    def _should_trigger_flash(self, context, best_move, best_flash):
        if context["visible_monster_count"] == 0:
            return False, "NO_VISIBLE_MONSTER"

        if context["in_threat"]:
            return True, "THREAT"

        if context["in_near_threat"]:
            move_good_enough = self._move_is_good_enough(best_move, context)
            flash_has_escape_value = self._has_escape_value(best_flash, context)
            flash_clearly_better = bool(
                best_flash is not None
                and best_move is not None
                and best_flash["score"] >= best_move["score"] + Config.FLASH_NEAR_TRIGGER_SCORE_MARGIN
            )
            if flash_has_escape_value and (not move_good_enough or flash_clearly_better):
                return True, "NEAR_THREAT"
            return False, "MOVE_GOOD_ENOUGH"

        return False, "NOT_IN_DANGER"

    def _build_action_bias(self, context, best_move, best_flash, decision_result):
        bias = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        flash_legal_actions = [action for action in range(8, 16) if context["legal_action"][action]]

        if decision_result["flash_execute"] and best_flash is not None:
            boost = max(
                Config.FLASH_OVERRIDE_MARGIN,
                best_flash["score"] - best_move["score"] + 1.0,
            )
            bias[best_flash["action"]] += Config.FLASH_POLICY_BIAS_SCALE * float(boost)
            if best_move is not None:
                bias[best_move["action"]] -= 0.5 * Config.FLASH_POLICY_MIN_SUPPRESS
            for action in flash_legal_actions:
                if action == best_flash["action"]:
                    continue
                bias[action] -= 0.5 * Config.FLASH_POLICY_MIN_SUPPRESS
            return bias.tolist()

        if flash_legal_actions:
            suppress = Config.FLASH_POLICY_MIN_SUPPRESS
            if not decision_result["flash_eval_triggered"]:
                suppress = max(suppress, Config.FLASH_NON_TRIGGER_SUPPRESS)
            if best_flash is not None and best_move is not None:
                suppress = max(suppress, best_move["score"] - best_flash["score"])
            for action in flash_legal_actions:
                bias[action] -= float(suppress)

        if best_move is not None:
            bias[best_move["action"]] += 0.5 * Config.FLASH_POLICY_BIAS_SCALE * Config.FLASH_OVERRIDE_MARGIN
        return bias.tolist()

    def _has_escape_value(self, candidate, context):
        if candidate is None or candidate["invalid"] or not candidate["legal"]:
            return False
        if candidate["leave_threat"]:
            return True
        if context["in_near_threat"] and candidate["leave_near_threat"]:
            return candidate["min_margin_gain_valid"] and candidate["min_margin_gain"] >= Config.FLASH_ESCAPE_MIN_MARGIN_GAIN
        if candidate["wall_cross"] or candidate["choke_escape"]:
            return (
                candidate["min_margin_gain_valid"]
                and candidate["min_margin_gain"] >= Config.FLASH_ESCAPE_MIN_MARGIN_GAIN
                and (
                    not candidate["distance_gain_valid"]
                    or candidate["distance_gain"] >= Config.FLASH_ESCAPE_DISTANCE_GAIN
                )
            )
        return (
            candidate["distance_gain_valid"]
            and candidate["min_margin_gain_valid"]
            and candidate["distance_gain"] >= Config.FLASH_ESCAPE_DISTANCE_GAIN
            and candidate["min_margin_gain"] >= Config.FLASH_ESCAPE_MIN_MARGIN_GAIN
        )

    def _move_is_good_enough(self, best_move, context):
        if best_move is None or best_move["invalid"] or not best_move["legal"]:
            return False
        if context["in_threat"]:
            return (
                best_move["leave_threat"]
                and best_move["min_margin_gain_valid"]
                and best_move["min_margin_gain"] >= Config.FLASH_DANGER_EFFECTIVE_MIN_MARGIN_GAIN
            )
        if context["in_near_threat"]:
            return (
                best_move["leave_near_threat"]
                and best_move["min_margin_gain_valid"]
                and best_move["min_margin_gain"] >= Config.FLASH_NEAR_TRIGGER_MIN_MARGIN_GAIN
            )
        return True

    def _best_candidate(self, candidates):
        if not candidates:
            return None
        legal_candidates = [candidate for candidate in candidates if candidate["legal"]]
        if not legal_candidates:
            return max(candidates, key=lambda item: item["score"])
        return max(legal_candidates, key=lambda item: item["score"])

    def _legacy_flash_reward(self, flash_effective, leave_threat, invalid_flash):
        if invalid_flash:
            return -0.02
        if flash_effective and leave_threat:
            return 0.02
        if flash_effective:
            return 0.01
        return 0.0

    def _empty_transition_info(self):
        return {
            "flash_used": False,
            "flash_effective": False,
            "flash": 0,
            "flash_in_danger": False,
            "flash_in_safe": False,
            "safe_flash": 0,
            "danger_flash": 0,
            "unknown_flash": 0,
            "danger_effective_flash": False,
            "danger_ineffective_flash": False,
            "escape_effective_flash": 0,
            "distance_gain_valid": False,
            "distance_gain": 0.0,
            "min_margin_gain_valid": False,
            "min_margin_gain": 0.0,
            "gain_unavailable_reason": "",
            "openness_gain": 0.0,
            "flash_distance_delta": 0.0,
            "flash_distance_delta_raw": 0.0,
            "flash_min_margin_gain": 0.0,
            "flash_openness_gain": 0.0,
            "flash_pre_in_threat": 0,
            "flash_pre_in_near_threat": 0,
            "leave_threat": False,
            "flash_leave_threat": 0,
            "leave_danger": False,
            "flash_leave_danger": 0,
            "post_flash_dead_end": False,
            "wall_cross": False,
            "choke_escape": False,
            "wall_cross_flash": 0,
            "choke_escape_flash": 0,
            "wall_cross_effective": 0,
            "choke_escape_effective": 0,
            "invalid_flash": 0,
            "escape_flash": 0,
            "flash_reward": 0.0,
            "early_flash": 0,
            "leave_near_threat": False,
        }
