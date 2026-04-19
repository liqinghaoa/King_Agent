#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and simple reward shaping for the DIY PPO stage-3A agent.
"""

import numpy as np

from agent_diy.conf.conf import Config

MAP_SIZE = 128.0
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
        if isinstance(legal_act_raw[0], bool):
            for j in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                legal_action[j] = int(legal_act_raw[j])
        else:
            valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < Config.ACTION_NUM}
            legal_action = [1 if j in valid_set else 0 for j in range(Config.ACTION_NUM)]

    if sum(legal_action) == 0:
        legal_action = [1] * Config.ACTION_NUM
    return legal_action


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_min_monster_raw_dist = None
        self.last_hero_pos = None
        self.last_in_danger = False
        self.last_in_semi_danger = False
        self.last_visible_monster = False
        self.last_flash_info = self._empty_flash_info()

    def _empty_flash_info(self):
        return {
            "flash_used": False,
            "flash_effective": False,
            "cur_min_dist_norm": 1.0,
            "prev_min_dist_norm": self.last_min_monster_dist_norm,
            "flash_distance_delta": 0.0,
            "flash_in_danger": False,
            "flash_in_safe": False,
            "flash": 0,
            "danger_flash": 0,
            "safe_flash": 0,
            "escape_flash": 0,
            "invalid_flash": 0,
            "flash_reward": 0.0,
        }

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)
        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_monster_raw_dist = None
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                    if cur_min_monster_raw_dist is None:
                        cur_min_monster_raw_dist = raw_dist
                    else:
                        cur_min_monster_raw_dist = min(cur_min_monster_raw_dist, raw_dist)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
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

        legal_action = _get_legal_action(observation)
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, step_norm], dtype=np.float32)

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        flash_reward, flash_info = self._calc_flash_reward(
            last_action=last_action,
            hero_pos=hero_pos,
            cur_min_monster_raw_dist=cur_min_monster_raw_dist,
            cur_min_dist_norm=cur_min_dist_norm,
        )

        reward = [survive_reward + dist_shaping + flash_reward]
        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_monster_raw_dist = cur_min_monster_raw_dist
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
        self.last_visible_monster = cur_min_monster_raw_dist is not None
        self.last_in_danger = cur_min_monster_raw_dist is not None and cur_min_monster_raw_dist <= 3.0
        self.last_in_semi_danger = cur_min_monster_raw_dist is not None and cur_min_monster_raw_dist <= 6.0
        self.last_flash_info = flash_info
        return feature, legal_action, reward

    def _calc_flash_reward(self, last_action, hero_pos, cur_min_monster_raw_dist, cur_min_dist_norm):
        flash_info = self._empty_flash_info()
        prev_min_dist_norm = self.last_min_monster_dist_norm
        flash_distance_delta = cur_min_dist_norm - prev_min_dist_norm
        flash_used = last_action >= 8

        flash_info.update(
            {
                "flash_used": bool(flash_used),
                "flash_effective": bool(flash_used and flash_distance_delta > 0.0),
                "cur_min_dist_norm": float(cur_min_dist_norm),
                "prev_min_dist_norm": float(prev_min_dist_norm),
                "flash_distance_delta": float(flash_distance_delta),
                "flash_in_danger": bool(flash_used and prev_min_dist_norm < 0.18),
                "flash_in_safe": bool(flash_used and prev_min_dist_norm > 0.45),
                "flash": int(flash_used),
            }
        )

        if last_action < 8:
            return 0.0, flash_info

        cur_visible_monster = cur_min_monster_raw_dist is not None
        cur_in_danger = cur_visible_monster and cur_min_monster_raw_dist <= 3.0
        cur_in_semi_danger = cur_visible_monster and cur_min_monster_raw_dist <= 6.0

        pos_delta = None
        if self.last_hero_pos is not None:
            pos_delta = np.sqrt(
                (hero_pos["x"] - self.last_hero_pos["x"]) ** 2
                + (hero_pos["z"] - self.last_hero_pos["z"]) ** 2
            )

        dist_delta = None
        if self.last_visible_monster and cur_visible_monster:
            dist_delta = cur_min_monster_raw_dist - self.last_min_monster_raw_dist

        flash_reward = 0.0
        last_near_danger = self.last_visible_monster and self.last_in_semi_danger
        if last_near_danger:
            flash_info["danger_flash"] = 1
        elif self.last_visible_monster:
            flash_info["safe_flash"] = 1

        if flash_info["flash_in_danger"]:
            flash_info["danger_flash"] = 1
        if flash_info["flash_in_safe"]:
            flash_info["safe_flash"] = 1

        if last_near_danger and dist_delta is not None and dist_delta > 1.0:
            flash_reward += 0.08

        if self.last_in_danger and not cur_in_danger:
            flash_reward += 0.12
            flash_info["escape_flash"] = 1

        invalid_flash = False
        if pos_delta is not None and pos_delta < 2.0:
            flash_reward -= 0.07
            invalid_flash = True

        if (
            self.last_visible_monster
            and cur_visible_monster
            and not self.last_in_semi_danger
            and not cur_in_semi_danger
            and (dist_delta is None or dist_delta <= 1.0)
        ):
            flash_reward -= 0.05
            invalid_flash = True

        if invalid_flash:
            flash_info["invalid_flash"] = 1

        flash_info["flash_reward"] = flash_reward
        return flash_reward, flash_info
