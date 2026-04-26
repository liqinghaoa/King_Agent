#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent class for recurrent discrete SAC.
"""

import json
import os
from collections import deque

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = device
        self.model = Model(device).to(self.device)
        self.algorithm = Algorithm(self.model, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        self.debug_eval = os.getenv("FLASH_RISK_DEBUG_EVAL", "0") == "1"
        self.eval_debug_history = deque(maxlen=8)
        self.actor_hidden_state = None
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1
        self.actor_hidden_state = None
        self.eval_debug_history.clear()

    def observation_process(self, env_obs):
        feature, temporal_feature, legal_action, reward = self.preprocessor.feature_process(
            env_obs, self.last_action
        )
        obs_data = ObsData(
            feature=list(feature),
            temporal_feature=list(temporal_feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        _, prob = self._run_policy(
            obs_data.feature,
            obs_data.temporal_feature,
            obs_data.legal_action,
            update_hidden=True,
        )
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)
        return [ActData(action=[action], d_action=[d_action], prob=list(prob))]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        _, prob = self._run_policy(
            obs_data.feature,
            obs_data.temporal_feature,
            obs_data.legal_action,
            update_hidden=True,
        )
        action = self._legal_sample(prob, use_max=True)
        self._log_eval_debug(env_obs, prob, action)
        act_data = ActData(action=[action], d_action=[action], prob=list(prob))
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def get_episode_metrics(self):
        if hasattr(self.preprocessor, "get_episode_metrics"):
            return self.preprocessor.get_episode_metrics()
        return {
            "stalled_move_rate": 0.0,
            "oscillation_alert_rate": 0.0,
            "effective_flash_rate": 0.0,
            "wasted_flash_rate": 0.0,
            "danger_flash_rate": 0.0,
            "safe_flash_rate": 0.0,
            "danger_effective_flash_rate": 0.0,
            "flash_eval_trigger_rate": 0.0,
            "best_flash_better_than_move_rate": 0.0,
            "no_flash_move_better_rate": 0.0,
            "close_escape_flash_rate": 0.0,
            "wall_cross_flash_rate": 0.0,
            "choke_escape_flash_rate": 0.0,
            "wall_cross_effective_rate": 0.0,
            "flash_pre_in_threat_rate": 0.0,
            "flash_pre_in_near_threat_rate": 0.0,
            "flash_leave_danger_rate": 0.0,
            "flash_leave_threat_rate": 0.0,
            "post_flash_dead_end_rate": 0.0,
            "avg_flash_distance_gain": 0.0,
            "avg_flash_min_margin_gain": 0.0,
            "avg_flash_openness_gain": 0.0,
            "flash_blocked_rate": 0.0,
            "revisit_rate": 0.0,
            "dead_end_entry_rate": 0.0,
            "dead_end_flash_escape_rate": 0.0,
            "dead_end_backtrack_rate": 0.0,
            "dead_end_local_mode_rate": 0.0,
            "dead_end_local_commit_rate": 0.0,
            "dead_end_flash_follow_rate": 0.0,
            "dead_end_backtrack_follow_rate": 0.0,
            "dead_end_local_follow_rate": 0.0,
            "post_flash_follow_rate": 0.0,
            "post_flash_pause_rate": 0.0,
            "discovery_step_rate": 0.0,
            "map_coverage_ratio": 0.0,
            "hidden_treasure_memory_rate": 0.0,
            "frontier_available_rate": 0.0,
            "frontier_follow_rate": 0.0,
            "loop_survival_mode_rate": 0.0,
            "loop_anchor_follow_rate": 0.0,
        }

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        checkpoint = self.algorithm.build_checkpoint()
        torch.save(checkpoint, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        checkpoint = torch.load(model_file_path, map_location=self.device)
        self.algorithm.load_checkpoint(checkpoint)
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_policy(self, feature, temporal_feature, legal_action, update_hidden):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32, device=self.device)
        temporal_tensor = torch.tensor(
            np.array([temporal_feature]),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits, next_hidden = self.model.policy(
                obs_tensor,
                temporal_tensor,
                hidden_state=self.actor_hidden_state,
                return_hidden=True,
            )
        if update_hidden:
            self.actor_hidden_state = self._detach_hidden(next_hidden)
        logits_np = logits.cpu().numpy()[0]
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)
        return logits_np, prob

    def _detach_hidden(self, hidden_state):
        if hidden_state is None:
            return None
        if isinstance(hidden_state, tuple):
            return tuple(item.detach() for item in hidden_state)
        return hidden_state.detach()

    def _legal_soft_max(self, logits, legal_action):
        weight = 1e20
        eps = 1e-5
        masked_logits = logits - weight * (1.0 - legal_action)
        masked_logits = np.clip(masked_logits - np.max(masked_logits, keepdims=True), -weight, 1)
        prob = (np.exp(masked_logits) + eps) * legal_action
        prob_sum = np.sum(prob, keepdims=True)
        if prob_sum <= 0:
            legal_count = np.sum(legal_action)
            if legal_count <= 0:
                return np.ones_like(prob) / float(len(prob))
            return legal_action / legal_count
        return prob / (prob_sum * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _log_eval_debug(self, env_obs, prob, action):
        if not self.debug_eval or not self.logger:
            return

        observation = env_obs.get("observation", {}) if isinstance(env_obs, dict) else {}
        frame_state = observation.get("frame_state", {}) if isinstance(observation, dict) else {}
        hero = frame_state.get("heroes", {}) if isinstance(frame_state, dict) else {}
        hero_pos = hero.get("pos", {}) if isinstance(hero, dict) else {}

        step_no = int(observation.get("step_no", -1)) if isinstance(observation, dict) else -1
        hero_x = int(hero_pos.get("x", -1)) if isinstance(hero_pos, dict) else -1
        hero_z = int(hero_pos.get("z", -1)) if isinstance(hero_pos, dict) else -1

        top_indices = np.argsort(prob)[-3:][::-1]
        top_probs = {int(idx): round(float(prob[idx]), 4) for idx in top_indices}

        debug_info = dict(getattr(self.preprocessor, "last_debug_info", {}) or {})
        if action >= 8:
            selected_flash_action = int(action % 8)
            flash_info_by_action = dict(getattr(self.preprocessor, "last_flash_info_by_action", {}) or {})
            selected_flash_info = dict(flash_info_by_action.get(selected_flash_action, {}) or {})
            if selected_flash_info:
                debug_info["selected_flash_action"] = selected_flash_action
                debug_info["selected_flash_landing_ratio"] = round(
                    float(selected_flash_info.get("landing_ratio", 0.0)), 4
                )
                debug_info["selected_flash_space_score"] = round(
                    float(selected_flash_info.get("landing_space_score", 0.0)), 4
                )
                debug_info["selected_flash_monster_gain"] = round(
                    float(selected_flash_info.get("monster_gain", 0.0)), 4
                )
                debug_info["selected_flash_soft_block"] = int(
                    bool(selected_flash_info.get("soft_block", False))
                )
                debug_info["selected_flash_escape_possible"] = int(
                    bool(selected_flash_info.get("escape_possible", False))
                )
        payload = {
            "kind": "eval_step",
            "step_no": step_no,
            "hero": [hero_x, hero_z],
            "action": int(action),
            "top_probs": top_probs,
            "memory": debug_info,
        }
        self.logger.info(f"[EVAL-TRACE] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}")
        if action >= 8:
            self.logger.warning(
                f"[EVAL-FLASH] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        if debug_info.get("dead_end_local_mode", 0) == 1 or debug_info.get("guidance_source") == "dead_end_local":
            self.logger.warning(
                f"[EVAL-LOCAL] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        if debug_info.get("dead_end_backtrack_mode", 0) == 1 or debug_info.get("guidance_source") == "dead_end_backtrack":
            self.logger.warning(
                f"[EVAL-BACK] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        if debug_info.get("flash_commit_mode", 0) == 1 or debug_info.get("guidance_source") == "flash_escape_commit":
            self.logger.warning(
                f"[EVAL-COMMIT] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        if debug_info.get("post_flash_follow_mode", 0) == 1 or debug_info.get("guidance_source") == "post_flash_follow":
            self.logger.warning(
                f"[EVAL-POST] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        if debug_info.get("flash_planner_override", 0) == 1 or debug_info.get("guidance_source") == "flash_planner":
            self.logger.warning(
                f"[EVAL-PLANNER] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        if debug_info.get("anti_oscillation_mode", 0) == 1 or debug_info.get("guidance_source") == "anti_oscillation":
            self.logger.warning(
                f"[EVAL-ANTI] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        if debug_info.get("dead_end_under_pressure", 0) == 1 or (
            debug_info.get("narrow_topology_flag", 0) == 1
            and float(debug_info.get("cur_min_monster_dist_norm", 1.0)) < 0.11
        ):
            self.logger.warning(
                f"[EVAL-NARROW] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )

        self.eval_debug_history.append(
            {
                "step_no": step_no,
                "hero": (hero_x, hero_z),
                "action": int(action),
            }
        )

        if len(self.eval_debug_history) >= 4:
            recent = list(self.eval_debug_history)[-4:]
            pos = [item["hero"] for item in recent]
            if pos[0] == pos[2] and pos[1] == pos[3] and pos[0] != pos[1]:
                self.logger.warning(
                    "[EVAL-OSC] "
                    + json.dumps(
                        {
                            "pattern": recent,
                            "memory": debug_info,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )

        if len(self.eval_debug_history) >= 6:
            recent = list(self.eval_debug_history)[-6:]
            unique_pos = sorted({item["hero"] for item in recent})
            if len(unique_pos) <= 2:
                self.logger.warning(
                    "[EVAL-STUCK] "
                    + json.dumps(
                        {
                            "recent": recent,
                            "unique_positions": unique_pos,
                            "memory": debug_info,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
