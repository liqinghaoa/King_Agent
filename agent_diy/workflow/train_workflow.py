#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for discrete SAC.
"""

import hashlib
import json
import os
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import TransitionData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


TRAIN_CONTROL_PATH = Path(__file__).resolve().parents[2] / "conf" / "train_control.toml"
DEFAULT_STOP_CHECK_INTERVAL_SECONDS = 10


def _extract_train_global_step(training_metrics):
    if not isinstance(training_metrics, dict):
        return None

    basic_metrics = training_metrics.get("basic")
    if isinstance(basic_metrics, dict) and "train_global_step" in basic_metrics:
        try:
            return int(float(basic_metrics["train_global_step"]))
        except (TypeError, ValueError):
            return None

    if "train_global_step" in training_metrics:
        try:
            return int(float(training_metrics["train_global_step"]))
        except (TypeError, ValueError):
            return None

    return None


def _load_train_control(logger=None):
    train_control = {
        "max_train_global_step": 0,
        "stop_check_interval_seconds": DEFAULT_STOP_CHECK_INTERVAL_SECONDS,
    }

    if not TRAIN_CONTROL_PATH.exists():
        return train_control

    try:
        with TRAIN_CONTROL_PATH.open("rb") as fh:
            config = tomllib.load(fh)
        section = config.get("train_control", {})
        train_control["max_train_global_step"] = max(0, int(section.get("max_train_global_step", 0) or 0))
        train_control["stop_check_interval_seconds"] = max(
            1,
            int(section.get("stop_check_interval_seconds", DEFAULT_STOP_CHECK_INTERVAL_SECONDS) or 0),
        )
        if logger and train_control["max_train_global_step"] > 0:
            logger.info(
                "train auto-stop enabled: "
                f"max_train_global_step={train_control['max_train_global_step']} "
                f"stop_check_interval_seconds={train_control['stop_check_interval_seconds']}"
            )
    except Exception as exc:  # pragma: no cover
        if logger:
            logger.warning(f"failed to load train control config {TRAIN_CONTROL_PATH}: {exc}")
    return train_control


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_map_profile(env_obs):
    observation = env_obs.get("observation", {}) if isinstance(env_obs, dict) else {}
    env_info = observation.get("env_info", {}) if isinstance(observation, dict) else {}
    frame_state = observation.get("frame_state", {}) if isinstance(observation, dict) else {}
    hero = frame_state.get("heroes", {}) if isinstance(frame_state, dict) else {}
    hero_pos = hero.get("pos", {}) if isinstance(hero, dict) else {}
    map_info = observation.get("map_info", []) if isinstance(observation, dict) else []
    organs = frame_state.get("organs", []) if isinstance(frame_state, dict) else []
    monsters = frame_state.get("monsters", []) if isinstance(frame_state, dict) else []

    map_id = None
    for key in ("map_id", "map_idx", "map_no", "map", "map_name"):
        if key in env_info:
            map_id = env_info.get(key)
            break

    start_open_ratio = 0.0
    start_cross_open_ratio = 0.0
    flattened_map = []
    if isinstance(map_info, list) and map_info and isinstance(map_info[0], list):
        flattened_map = [int(cell) for row in map_info for cell in row]
        if flattened_map:
            start_open_ratio = float(sum(flattened_map)) / float(len(flattened_map))
        center = len(map_info) // 2
        cross_cells = []
        for idx in range(len(map_info)):
            cross_cells.append(int(map_info[center][idx]))
            if idx != center:
                cross_cells.append(int(map_info[idx][center]))
        if cross_cells:
            start_cross_open_ratio = float(sum(cross_cells)) / float(len(cross_cells))

    visible_treasures = []
    visible_buffs = []
    for organ in organs if isinstance(organs, list) else []:
        if organ.get("status", 1) != 1:
            continue
        organ_info = {
            "config_id": organ.get("config_id", -1),
            "dir": organ.get("hero_relative_direction", -1),
            "dist": organ.get("hero_l2_distance", -1),
        }
        if organ.get("sub_type", 0) == 1:
            visible_treasures.append(organ_info)
        elif organ.get("sub_type", 0) == 2:
            visible_buffs.append(organ_info)

    monster_dist_buckets = [
        _safe_float(monster.get("hero_l2_distance", 5.0), 5.0)
        for monster in monsters
        if isinstance(monster, dict)
    ]
    start_min_monster_dist_bucket = min(monster_dist_buckets) if monster_dist_buckets else 5.0

    signature_payload = {
        "map_id": map_id,
        "hero_x": hero_pos.get("x"),
        "hero_z": hero_pos.get("z"),
        "map_info": flattened_map,
        "treasures": sorted(
            (item["config_id"], item["dir"], item["dist"]) for item in visible_treasures
        ),
        "buffs": sorted(
            (item["config_id"], item["dir"], item["dist"]) for item in visible_buffs
        ),
    }
    map_signature = hashlib.md5(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]

    return {
        "map_id": map_id if map_id is not None else "na",
        "map_signature": map_signature,
        "hero_x": _safe_int(hero_pos.get("x", -1), -1),
        "hero_z": _safe_int(hero_pos.get("z", -1), -1),
        "start_open_ratio": start_open_ratio,
        "start_cross_open_ratio": start_cross_open_ratio,
        "start_visible_treasures": len(visible_treasures),
        "start_visible_buffs": len(visible_buffs),
        "start_min_monster_dist_bucket": start_min_monster_dist_bucket,
    }


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    last_stop_check_time = 0.0
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    if logger:
        logger.info(
            "[RNN-CONFIG] "
            + json.dumps(
                {
                    "use_recurrent": bool(Config.USE_RECURRENT),
                    "seq_len": int(Config.SEQ_LEN),
                    "burn_in": int(Config.BURN_IN),
                    "learn_len": int(Config.LEARN_LEN),
                    "lstm_hidden_dim": int(Config.LSTM_HIDDEN_DIM),
                    "lstm_num_layers": int(Config.LSTM_NUM_LAYERS),
                    "use_gru": bool(Config.USE_GRU),
                    "static_hidden_dim": int(Config.STATIC_HIDDEN_DIM),
                    "dynamic_hidden_dim": int(Config.DYNAMIC_HIDDEN_DIM),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )
    train_control = _load_train_control(logger)

    while True:
        for game_data in episode_runner.run_episodes():
            agent.send_sample_data(game_data)
            game_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now

            if (
                train_control["max_train_global_step"] > 0
                and now - last_stop_check_time >= train_control["stop_check_interval_seconds"]
            ):
                training_metrics = get_training_metrics()
                last_stop_check_time = now
                current_train_global_step = _extract_train_global_step(training_metrics)
                if current_train_global_step is None:
                    continue

                if current_train_global_step >= train_control["max_train_global_step"]:
                    logger.info(
                        "train auto-stop triggered: "
                        f"train_global_step={current_train_global_step} "
                        f"target={train_control['max_train_global_step']}"
                    )
                    agent.save_model()
                    logger.info("workflow exit after reaching configured max_train_global_step")
                    return


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0.0
        self.last_get_training_metrics_time = 0.0

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            map_profile = _extract_map_profile(env_obs)

            obs_data, _ = self.agent.observation_process(env_obs)
            collector = []
            self.episode_cnt += 1
            step = 0
            total_reward = 0.0

            self.logger.info(
                f"Episode {self.episode_cnt} start "
                f"map_id:{map_profile['map_id']} map_sig:{map_profile['map_signature']} "
                f"start_open:{map_profile['start_open_ratio']:.3f} "
                f"start_cross_open:{map_profile['start_cross_open_ratio']:.3f} "
                f"start_visible_treasures:{map_profile['start_visible_treasures']} "
                f"start_visible_buffs:{map_profile['start_visible_buffs']} "
                f"start_min_monster_dist:{map_profile['start_min_monster_dist_bucket']:.2f} "
                f"start_pos:({map_profile['hero_x']},{map_profile['hero_z']})"
            )

            while True:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                _, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                done = terminated or truncated
                step += 1

                next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
                reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                frame = TransitionData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    temporal_obs=np.array(obs_data.temporal_feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.int64),
                    reward=reward,
                    next_obs=np.array(next_obs_data.feature, dtype=np.float32),
                    next_temporal_obs=np.array(next_obs_data.temporal_feature, dtype=np.float32),
                    next_legal_action=np.array(next_obs_data.legal_action, dtype=np.float32),
                    done=np.array([float(done)], dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0.0)
                    treasures_collected = env_info.get("treasures_collected", 0)
                    collected_buff = env_info.get("collected_buff", 0)
                    flash_count = env_info.get("flash_count", 0)
                    episode_metrics = self.agent.get_episode_metrics()
                    temporal_summary = self.agent.get_temporal_summary()

                    final_reward = np.array([-10.0 if terminated else 10.0], dtype=np.float32)
                    collector[-1].reward = collector[-1].reward + final_reward

                    result_str = "FAIL" if terminated else "WIN"
                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"map_id:{map_profile['map_id']} map_sig:{map_profile['map_signature']} "
                        f"start_open:{map_profile['start_open_ratio']:.3f} "
                        f"start_visible_treasures:{map_profile['start_visible_treasures']} "
                        f"start_visible_buffs:{map_profile['start_visible_buffs']} "
                        f"start_min_monster_dist:{map_profile['start_min_monster_dist_bucket']:.2f} "
                        f"result:{result_str} sim_score:{float(total_score):.1f} "
                        f"treasures:{int(treasures_collected)} buffs:{int(collected_buff)} "
                        f"flash:{int(flash_count)} total_reward:{total_reward:.3f} "
                        f"flash_execute_count:{int(episode_metrics['flash_execute_count'])} "
                        f"flash_action_count:{int(episode_metrics['flash_action_count'])} "
                        f"danger_flash_count:{int(episode_metrics['danger_flash_count'])} "
                        f"safe_flash_count:{int(episode_metrics['safe_flash_count'])} "
                        f"effective_flash_count:{int(episode_metrics['effective_flash_count'])} "
                        f"danger_effective_flash_count:{int(episode_metrics['danger_effective_flash_count'])} "
                        f"flash_eval_trigger_count:{int(episode_metrics['flash_eval_trigger_count'])} "
                        f"flash_leave_danger_count:{int(episode_metrics['flash_leave_danger_count'])} "
                        f"flash_leave_threat_count:{int(episode_metrics['flash_leave_threat_count'])} "
                        f"wall_cross_flash_count:{int(episode_metrics['wall_cross_flash_count'])} "
                        f"wall_cross_effective_count:{int(episode_metrics['wall_cross_effective_count'])} "
                        f"choke_escape_flash_count:{int(episode_metrics['choke_escape_flash_count'])} "
                        f"choke_escape_effective_count:{int(episode_metrics['choke_escape_effective_count'])} "
                        f"stalled_rate:{episode_metrics['stalled_move_rate']:.3f} "
                        f"oscillation_alert_rate:{episode_metrics['oscillation_alert_rate']:.3f} "
                        f"effective_flash_rate:{episode_metrics['effective_flash_rate']:.3f} "
                        f"wasted_flash_rate:{episode_metrics['wasted_flash_rate']:.3f} "
                        f"danger_flash_rate:{episode_metrics['danger_flash_rate']:.3f} "
                        f"safe_flash_rate:{episode_metrics['safe_flash_rate']:.3f} "
                        f"danger_effective_flash_rate:{episode_metrics['danger_effective_flash_rate']:.3f} "
                        f"flash_eval_trigger_rate:{episode_metrics['flash_eval_trigger_rate']:.3f} "
                        f"best_flash_better_than_move_rate:{episode_metrics['best_flash_better_than_move_rate']:.3f} "
                        f"flash_leave_danger_rate:{episode_metrics['flash_leave_danger_rate']:.3f} "
                        f"wall_cross_effective_rate:{episode_metrics['wall_cross_effective_rate']:.3f} "
                        f"avg_flash_distance_gain:{episode_metrics['avg_flash_distance_gain']:.3f} "
                        f"avg_flash_min_margin_gain:{episode_metrics['avg_flash_min_margin_gain']:.3f} "
                        f"avg_flash_openness_gain:{episode_metrics['avg_flash_openness_gain']:.3f} "
                        f"flash_blocked_rate:{episode_metrics['flash_blocked_rate']:.3f} "
                        f"revisit_rate:{episode_metrics['revisit_rate']:.3f} "
                        f"dead_end_entry_rate:{episode_metrics['dead_end_entry_rate']:.3f} "
                        f"dead_end_flash_escape_rate:{episode_metrics['dead_end_flash_escape_rate']:.3f} "
                        f"dead_end_local_mode_rate:{episode_metrics['dead_end_local_mode_rate']:.3f} "
                        f"dead_end_local_commit_rate:{episode_metrics['dead_end_local_commit_rate']:.3f} "
                        f"dead_end_flash_follow_rate:{episode_metrics['dead_end_flash_follow_rate']:.3f} "
                        f"dead_end_local_follow_rate:{episode_metrics['dead_end_local_follow_rate']:.3f} "
                        f"dead_end_exit_success_rate:{episode_metrics['dead_end_exit_success_rate']:.3f} "
                        f"dead_end_reverse_follow_rate:{episode_metrics['dead_end_reverse_follow_rate']:.3f} "
                        f"persistent_dead_end_follow_rate:{episode_metrics['persistent_dead_end_follow_rate']:.3f} "
                        f"persistent_dead_end_active_rate:{episode_metrics['persistent_dead_end_active_rate']:.3f} "
                        f"persistent_dead_end_commit_rate:{episode_metrics['persistent_dead_end_commit_rate']:.3f} "
                        f"persistent_dead_end_success_follow_rate:{episode_metrics['persistent_dead_end_success_follow_rate']:.3f} "
                        f"dead_end_pretrigger_rate:{episode_metrics['dead_end_pretrigger_rate']:.3f} "
                        f"dead_end_deeper_block_rate:{episode_metrics['dead_end_deeper_block_rate']:.3f} "
                        f"confirmed_dead_end_rate:{episode_metrics['confirmed_dead_end_rate']:.3f} "
                        f"dead_end_reentry_block_rate:{episode_metrics['dead_end_reentry_block_rate']:.3f} "
                        f"discovery_step_rate:{episode_metrics['discovery_step_rate']:.3f} "
                        f"map_coverage_ratio:{episode_metrics['map_coverage_ratio']:.3f} "
                        f"hidden_treasure_memory_rate:{episode_metrics['hidden_treasure_memory_rate']:.3f} "
                        f"frontier_available_rate:{episode_metrics['frontier_available_rate']:.3f} "
                        f"frontier_follow_rate:{episode_metrics['frontier_follow_rate']:.3f} "
                        f"loop_survival_mode_rate:{episode_metrics['loop_survival_mode_rate']:.3f} "
                        f"loop_anchor_follow_rate:{episode_metrics['loop_anchor_follow_rate']:.3f}"
                    )

                    if (
                        self.logger
                        and Config.TEMPORAL_SUMMARY_LOG
                        and self.episode_cnt % max(1, int(Config.TEMPORAL_SUMMARY_INTERVAL)) == 0
                    ):
                        temporal_payload = {
                            "episode": int(self.episode_cnt),
                            "steps": int(step),
                            "result": result_str,
                            "map_id": map_profile["map_id"],
                            "map_signature": map_profile["map_signature"],
                        }
                        if isinstance(temporal_summary, dict):
                            temporal_payload.update(temporal_summary)
                        self.logger.info(
                            "[TEMPORAL-SUMMARY] "
                            + json.dumps(temporal_payload, ensure_ascii=True, sort_keys=True)
                        )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "sim_score": round(float(total_score), 4),
                            "treasures_collected": int(treasures_collected),
                            "collected_buff": int(collected_buff),
                            "flash_count": int(flash_count),
                            "flash_execute_count": int(episode_metrics["flash_execute_count"]),
                            "flash_action_count": int(episode_metrics["flash_action_count"]),
                            "danger_flash_count": int(episode_metrics["danger_flash_count"]),
                            "safe_flash_count": int(episode_metrics["safe_flash_count"]),
                            "effective_flash_count": int(episode_metrics["effective_flash_count"]),
                            "danger_effective_flash_count": int(
                                episode_metrics["danger_effective_flash_count"]
                            ),
                            "flash_eval_trigger_count": int(
                                episode_metrics["flash_eval_trigger_count"]
                            ),
                            "flash_leave_danger_count": int(
                                episode_metrics["flash_leave_danger_count"]
                            ),
                            "flash_leave_threat_count": int(
                                episode_metrics["flash_leave_threat_count"]
                            ),
                            "wall_cross_flash_count": int(
                                episode_metrics["wall_cross_flash_count"]
                            ),
                            "wall_cross_effective_count": int(
                                episode_metrics["wall_cross_effective_count"]
                            ),
                            "choke_escape_flash_count": int(
                                episode_metrics["choke_escape_flash_count"]
                            ),
                            "choke_escape_effective_count": int(
                                episode_metrics["choke_escape_effective_count"]
                            ),
                            "win_rate": 0.0 if terminated else 1.0,
                            "map_start_open_ratio": round(float(map_profile["start_open_ratio"]), 4),
                            "map_start_visible_treasures": int(map_profile["start_visible_treasures"]),
                            "map_start_visible_buffs": int(map_profile["start_visible_buffs"]),
                            "map_start_min_monster_dist": round(
                                float(map_profile["start_min_monster_dist_bucket"]), 4
                            ),
                            "revisit_rate": round(float(episode_metrics["revisit_rate"]), 4),
                            "dead_end_entry_rate": round(
                                float(episode_metrics["dead_end_entry_rate"]), 4
                            ),
                            "discovery_step_rate": round(
                                float(episode_metrics["discovery_step_rate"]), 4
                            ),
                            "map_coverage_ratio": round(
                                float(episode_metrics["map_coverage_ratio"]), 4
                            ),
                            "hidden_treasure_memory_rate": round(
                                float(episode_metrics["hidden_treasure_memory_rate"]), 4
                            ),
                            "frontier_available_rate": round(
                                float(episode_metrics["frontier_available_rate"]), 4
                            ),
                            "frontier_follow_rate": round(
                                float(episode_metrics["frontier_follow_rate"]), 4
                            ),
                            "loop_survival_mode_rate": round(
                                float(episode_metrics["loop_survival_mode_rate"]), 4
                            ),
                            "loop_anchor_follow_rate": round(
                                float(episode_metrics["loop_anchor_follow_rate"]), 4
                            ),
                            "stalled_move_rate": round(float(episode_metrics["stalled_move_rate"]), 4),
                            "oscillation_alert_rate": round(
                                float(episode_metrics["oscillation_alert_rate"]), 4
                            ),
                            "effective_flash_rate": round(
                                float(episode_metrics["effective_flash_rate"]), 4
                            ),
                            "wasted_flash_rate": round(float(episode_metrics["wasted_flash_rate"]), 4),
                            "danger_flash_rate": round(float(episode_metrics["danger_flash_rate"]), 4),
                            "safe_flash_rate": round(float(episode_metrics["safe_flash_rate"]), 4),
                            "danger_effective_flash_rate": round(
                                float(episode_metrics["danger_effective_flash_rate"]),
                                4,
                            ),
                            "flash_eval_trigger_rate": round(
                                float(episode_metrics["flash_eval_trigger_rate"]),
                                4,
                            ),
                            "best_flash_better_than_move_rate": round(
                                float(episode_metrics["best_flash_better_than_move_rate"]),
                                4,
                            ),
                            "no_flash_move_better_rate": round(
                                float(episode_metrics["no_flash_move_better_rate"]),
                                4,
                            ),
                            "flash_pre_in_threat_rate": round(
                                float(episode_metrics["flash_pre_in_threat_rate"]),
                                4,
                            ),
                            "flash_pre_in_near_threat_rate": round(
                                float(episode_metrics["flash_pre_in_near_threat_rate"]),
                                4,
                            ),
                            "flash_leave_danger_rate": round(
                                float(episode_metrics["flash_leave_danger_rate"]),
                                4,
                            ),
                            "flash_leave_threat_rate": round(
                                float(episode_metrics["flash_leave_threat_rate"]),
                                4,
                            ),
                            "wall_cross_flash_rate": round(
                                float(episode_metrics["wall_cross_flash_rate"]),
                                4,
                            ),
                            "wall_cross_effective_rate": round(
                                float(episode_metrics["wall_cross_effective_rate"]),
                                4,
                            ),
                            "avg_flash_distance_gain": round(
                                float(episode_metrics["avg_flash_distance_gain"]),
                                4,
                            ),
                            "avg_flash_min_margin_gain": round(
                                float(episode_metrics["avg_flash_min_margin_gain"]),
                                4,
                            ),
                            "avg_flash_openness_gain": round(
                                float(episode_metrics["avg_flash_openness_gain"]),
                                4,
                            ),
                            "post_flash_dead_end_rate": round(
                                float(episode_metrics["post_flash_dead_end_rate"]),
                                4,
                            ),
                            "flash_blocked_rate": round(
                                float(episode_metrics["flash_blocked_rate"]), 4
                            ),
                            "dead_end_flash_escape_rate": round(
                                float(episode_metrics["dead_end_flash_escape_rate"]), 4
                            ),
                            "dead_end_local_mode_rate": round(
                                float(episode_metrics["dead_end_local_mode_rate"]), 4
                            ),
                            "dead_end_local_commit_rate": round(
                                float(episode_metrics["dead_end_local_commit_rate"]), 4
                            ),
                            "dead_end_flash_follow_rate": round(
                                float(episode_metrics["dead_end_flash_follow_rate"]), 4
                            ),
                            "dead_end_local_follow_rate": round(
                                float(episode_metrics["dead_end_local_follow_rate"]), 4
                            ),
                            "dead_end_exit_success_rate": round(
                                float(episode_metrics["dead_end_exit_success_rate"]), 4
                            ),
                            "dead_end_reverse_follow_rate": round(
                                float(episode_metrics["dead_end_reverse_follow_rate"]), 4
                            ),
                            "persistent_dead_end_follow_rate": round(
                                float(episode_metrics["persistent_dead_end_follow_rate"]), 4
                            ),
                            "persistent_dead_end_active_rate": round(
                                float(episode_metrics["persistent_dead_end_active_rate"]), 4
                            ),
                            "persistent_dead_end_commit_rate": round(
                                float(episode_metrics["persistent_dead_end_commit_rate"]), 4
                            ),
                            "persistent_dead_end_success_follow_rate": round(
                                float(episode_metrics["persistent_dead_end_success_follow_rate"]), 4
                            ),
                            "dead_end_pretrigger_rate": round(
                                float(episode_metrics["dead_end_pretrigger_rate"]), 4
                            ),
                            "dead_end_deeper_block_rate": round(
                                float(episode_metrics["dead_end_deeper_block_rate"]), 4
                            ),
                            "confirmed_dead_end_rate": round(
                                float(episode_metrics["confirmed_dead_end_rate"]), 4
                            ),
                            "dead_end_reentry_block_rate": round(
                                float(episode_metrics["dead_end_reentry_block_rate"]), 4
                            ),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        yield sample_process(collector)
                    break

                obs_data = next_obs_data
