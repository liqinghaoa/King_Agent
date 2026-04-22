#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and planner context builder for the DIY PPO agent.
"""

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.flash_escape_strategy import FlashEscapeStrategy


MAP_SIZE = 128.0
MAP_DIAGONAL = MAP_SIZE * 1.41
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _get_legal_action(observation):
    legal_act_raw = observation.get("legal_action", observation.get("legal_act", None))
    legal_action = [1] * Config.ACTION_NUM

    if isinstance(legal_act_raw, np.ndarray):
        legal_act_raw = legal_act_raw.tolist()

    if isinstance(legal_act_raw, (list, tuple)) and legal_act_raw:
        is_full_binary_mask = len(legal_act_raw) >= Config.ACTION_NUM and all(
            int(a) in (0, 1) for a in legal_act_raw[: Config.ACTION_NUM]
        )
        if isinstance(legal_act_raw[0], bool) or is_full_binary_mask:
            for idx in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                legal_action[idx] = int(legal_act_raw[idx])
        else:
            valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < Config.ACTION_NUM}
            legal_action = [1 if idx in valid_set else 0 for idx in range(Config.ACTION_NUM)]

    if sum(legal_action) == 0:
        legal_action = [1] * Config.ACTION_NUM
    return legal_action


class Preprocessor:
    def __init__(self):
        self.flash_strategy = FlashEscapeStrategy()
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.current_planner_context = None
        self.last_policy_decision = None
        self.last_decision_info = self._empty_decision_info()
        self.last_flash_info = self._empty_flash_info()

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_action = _get_legal_action(observation)

        self.step_no = int(observation.get("step_no", 0))
        self.max_step = int(env_info.get("max_step", 200))

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_feat = np.array(
            [
                _norm(hero_pos["x"], MAP_SIZE),
                _norm(hero_pos["z"], MAP_SIZE),
                _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD),
                _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION),
            ],
            dtype=np.float32,
        )

        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for idx in range(2):
            if idx < len(monsters):
                monster = monsters[idx]
                is_in_view = float(monster.get("is_in_view", 0))
                monster_pos = monster.get("pos", {})
                if is_in_view and isinstance(monster_pos, dict):
                    raw_dist = self._distance(hero_pos, monster_pos) or MAP_DIAGONAL
                    dist_norm = _norm(raw_dist, MAP_DIAGONAL)
                    monster_feats.append(
                        np.array(
                            [
                                is_in_view,
                                _norm(monster_pos.get("x", 0.0), MAP_SIZE),
                                _norm(monster_pos.get("z", 0.0), MAP_SIZE),
                                _norm(monster.get("speed", 1), MAX_MONSTER_SPEED),
                                dist_norm,
                            ],
                            dtype=np.float32,
                        )
                    )
                else:
                    monster_feats.append(np.zeros(5, dtype=np.float32))
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        step_norm = _norm(self.step_no, self.max_step)
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                np.array([step_norm, step_norm], dtype=np.float32),
            ]
        )

        current_context = self.flash_strategy.build_context(observation, legal_action)
        transition_info = self._build_transition_info(last_action, current_context)

        survive_reward = 0.01
        dist_shaping = 0.1 * (self._current_min_dist_norm(current_context) - self.last_min_monster_dist_norm)
        flash_reward = float(transition_info.get("flash_reward", 0.0))
        if Config.DISABLE_LEGACY_FLASH_REWARD:
            flash_reward = 0.0
            transition_info["flash_reward"] = 0.0
        reward = [survive_reward + dist_shaping + flash_reward]

        self.last_min_monster_dist_norm = self._current_min_dist_norm(current_context)
        self.current_planner_context = current_context
        self.last_flash_info = transition_info

        return feature, legal_action, reward

    def get_planner_context(self):
        return self.current_planner_context

    def plan_flash_escape(self):
        if not Config.ENABLE_FLASH_ESCAPE_V1 or self.current_planner_context is None:
            return self._empty_planner_result()
        return self.flash_strategy.evaluate(self.current_planner_context)

    def record_policy_decision(self, planner_result, selected_action, deterministic_action, policy_bias):
        planner_result = planner_result or self._empty_planner_result()
        selected_action = int(selected_action)
        deterministic_action = int(deterministic_action)
        decision_info = self._sanitize_planner_result(planner_result)
        decision_info.update(
            {
                "planner_chosen_action": int(planner_result.get("chosen_action", -1)),
                "planner_chosen_action_type": str(planner_result.get("chosen_action_type", "move")),
                "selected_action": selected_action,
                "selected_action_type": "flash" if selected_action >= 8 else "move",
                "deterministic_action": deterministic_action,
                "deterministic_action_type": "flash" if deterministic_action >= 8 else "move",
                "policy_bias_norm": float(np.linalg.norm(np.array(policy_bias, dtype=np.float32), ord=1)),
            }
        )
        self.last_decision_info = decision_info
        self.last_policy_decision = {
            "context": self.current_planner_context,
            "planner_result": planner_result,
            "decision_info": decision_info,
        }

    def _build_transition_info(self, last_action, current_context):
        if last_action < 0 or self.last_policy_decision is None or self.last_policy_decision.get("context") is None:
            self.last_decision_info = self._empty_decision_info()
            return self._empty_flash_info()

        transition_info = self.flash_strategy.evaluate_transition(
            prev_context=self.last_policy_decision["context"],
            decision_record=self.last_policy_decision,
            action=last_action,
            next_context=current_context,
        )
        transition_info = self._normalize_flash_monitor_info(transition_info)
        self.last_decision_info = transition_info
        return transition_info

    def _current_min_dist_norm(self, context):
        min_euclid = float(context.get("min_euclid", MAP_DIAGONAL))
        if min_euclid >= 1e5:
            return 1.0
        return _norm(min_euclid, MAP_DIAGONAL)

    def _sanitize_planner_result(self, planner_result):
        return {
            "flash_eval_triggered": bool(planner_result.get("flash_eval_triggered", False)),
            "flash_execute": bool(planner_result.get("flash_execute", False)),
            "chosen_action": int(planner_result.get("chosen_action", -1)),
            "chosen_action_type": str(planner_result.get("chosen_action_type", "move")),
            "best_move_action": int(planner_result.get("best_move_action", -1)),
            "best_move_score": float(planner_result.get("best_move_score", 0.0)),
            "best_flash_action": int(planner_result.get("best_flash_action", -1)),
            "best_flash_score": float(planner_result.get("best_flash_score", 0.0)),
            "flash_skip_reason": str(planner_result.get("flash_skip_reason", "")),
            "wall_cross": bool(planner_result.get("wall_cross", False)),
            "choke_escape": bool(planner_result.get("choke_escape", False)),
            "leave_threat": bool(planner_result.get("leave_threat", False)),
            "distance_gain": float(planner_result.get("distance_gain", 0.0)),
            "min_margin_gain": float(planner_result.get("min_margin_gain", 0.0)),
            "openness_gain": float(planner_result.get("openness_gain", 0.0)),
            "post_flash_dead_end": bool(planner_result.get("post_flash_dead_end", False)),
            "early_flash": bool(planner_result.get("early_flash", False)),
            "decision_tag": str(planner_result.get("decision_tag", "MOVE_BETTER")),
            "best_flash_better_than_move": bool(planner_result.get("best_flash_better_than_move", False)),
            "best_move_better_than_flash": bool(planner_result.get("best_move_better_than_flash", True)),
            "trigger_reason": str(planner_result.get("trigger_reason", "")),
        }

    def _empty_planner_result(self):
        return {
            "flash_eval_triggered": False,
            "flash_execute": False,
            "chosen_action": -1,
            "chosen_action_type": "move",
            "best_move_action": -1,
            "best_move_score": 0.0,
            "best_flash_action": -1,
            "best_flash_score": 0.0,
            "flash_skip_reason": "NOT_TRIGGERED",
            "wall_cross": False,
            "choke_escape": False,
            "leave_threat": False,
            "distance_gain": 0.0,
            "min_margin_gain": 0.0,
            "openness_gain": 0.0,
            "post_flash_dead_end": False,
            "early_flash": False,
            "decision_tag": "MOVE_BETTER",
            "best_flash_better_than_move": False,
            "best_move_better_than_flash": True,
            "trigger_reason": "DISABLED",
            "action_bias": [0.0] * Config.ACTION_NUM,
            "_candidate_lookup": {},
        }

    def _empty_decision_info(self):
        return {
            "flash_eval_triggered": False,
            "flash_execute": False,
            "chosen_action": -1,
            "chosen_action_type": "move",
            "best_move_action": -1,
            "best_move_score": 0.0,
            "best_flash_action": -1,
            "best_flash_score": 0.0,
            "flash_skip_reason": "NOT_TRIGGERED",
            "wall_cross": False,
            "choke_escape": False,
            "leave_threat": False,
            "distance_gain": 0.0,
            "min_margin_gain": 0.0,
            "openness_gain": 0.0,
            "post_flash_dead_end": False,
            "early_flash": False,
            "decision_tag": "MOVE_BETTER",
            "best_flash_better_than_move": False,
            "best_move_better_than_flash": True,
            "trigger_reason": "",
        }

    def _empty_flash_info(self):
        info = self._empty_decision_info()
        info.update(
            {
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
                "non_escape_effective_flash": 0,
                "flash_distance_delta": 0.0,
                "flash_distance_delta_raw": 0.0,
                "flash_min_margin_gain": 0.0,
                "flash_openness_gain": 0.0,
                "flash_pre_in_threat": 0,
                "flash_pre_in_near_threat": 0,
                "flash_leave_threat": 0,
                "flash_leave_danger": 0,
                "wall_cross_flash": 0,
                "choke_escape_flash": 0,
                "wall_cross_effective": 0,
                "choke_escape_effective": 0,
                "invalid_flash": 0,
                "escape_flash": 0,
                "flash_reward": 0.0,
                "selected_action": -1,
                "selected_action_type": "move",
            }
        )
        return info

    def _normalize_flash_monitor_info(self, transition_info):
        info = dict(transition_info or {})
        flash_used = bool(info.get("flash_used", False))
        flash_effective = bool(info.get("flash_effective", False))
        leave_threat = bool(info.get("flash_leave_threat", 0))

        if not flash_used:
            info["flash_in_danger"] = False
            info["flash_in_safe"] = False
            info["danger_flash"] = 0
            info["safe_flash"] = 0
            info["unknown_flash"] = 0
            info["danger_effective_flash"] = False
            info["danger_ineffective_flash"] = False
            info["escape_effective_flash"] = 0
            info["non_escape_effective_flash"] = 0
            info["flash_pre_in_threat"] = 0
            info["flash_pre_in_near_threat"] = 0
            info["flash_leave_threat"] = 0
            info["flash_leave_danger"] = 0
            info["distance_gain"] = 0.0
            info["flash_distance_delta"] = 0.0
            info["min_margin_gain"] = 0.0
            info["flash_min_margin_gain"] = 0.0
            return info

        in_danger = bool(info.get("flash_in_danger", False))
        in_safe = bool(info.get("flash_in_safe", False))
        in_unknown = bool(info.get("unknown_flash", 0))

        if in_danger:
            flash_class = "danger"
        elif in_safe:
            flash_class = "safe"
        elif in_unknown:
            flash_class = "unknown"
        else:
            flash_class = "unknown"

        info["flash_in_danger"] = flash_class == "danger"
        info["flash_in_safe"] = flash_class == "safe"
        info["danger_flash"] = int(flash_class == "danger")
        info["safe_flash"] = int(flash_class == "safe")
        info["unknown_flash"] = int(flash_class == "unknown")

        if not bool(info.get("distance_gain_valid", True)):
            info["distance_gain"] = 0.0
            info["flash_distance_delta"] = 0.0
        if not bool(info.get("min_margin_gain_valid", True)):
            info["min_margin_gain"] = 0.0
            info["flash_min_margin_gain"] = 0.0

        danger_effective = bool(info.get("danger_effective_flash", False))
        if info["flash_in_danger"] and int(info.get("escape_effective_flash", 0)) > 0:
            danger_effective = True
        info["danger_effective_flash"] = bool(info["flash_in_danger"] and danger_effective)
        info["danger_ineffective_flash"] = bool(info["flash_in_danger"] and not info["danger_effective_flash"])

        escape_effective = int(info.get("escape_effective_flash", 0))
        if escape_effective <= 0 and flash_effective and leave_threat:
            escape_effective = 1
        non_escape_effective = int(flash_effective) - escape_effective
        info["flash_pre_in_threat"] = int(bool(info.get("flash_pre_in_threat", 0)))
        info["flash_pre_in_near_threat"] = int(bool(info.get("flash_pre_in_near_threat", 0)))
        info["flash_leave_threat"] = int(bool(info.get("flash_leave_threat", 0)))
        info["flash_leave_danger"] = int(bool(info.get("flash_leave_danger", 0)))
        info["escape_effective_flash"] = escape_effective
        info["non_escape_effective_flash"] = max(non_escape_effective, 0)
        return info

    def _distance(self, pos_a, pos_b):
        if not pos_a or not pos_b:
            return None
        try:
            return float(np.sqrt((pos_a["x"] - pos_b["x"]) ** 2 + (pos_a["z"] - pos_b["z"]) ** 2))
        except (KeyError, TypeError):
            return None
