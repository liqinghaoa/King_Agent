#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for the DIY PPO agent.
"""

import os
import time

import numpy as np

from agent_diy.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None and self.logger:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            flash_count = 0
            effective_flash_count = 0
            ineffective_flash_count = 0
            escape_effective_flash_count = 0
            non_escape_effective_flash_count = 0
            danger_flash_count = 0
            safe_flash_count = 0
            unknown_flash_count = 0
            danger_effective_flash_count = 0
            danger_ineffective_flash_count = 0
            escape_flash_count = 0
            invalid_flash_count = 0
            flash_pre_in_threat_count = 0
            flash_pre_in_near_threat_count = 0
            flash_leave_threat_count = 0
            flash_leave_danger_count = 0
            post_flash_dead_end_count = 0
            wall_cross_flash_count = 0
            wall_cross_effective_count = 0
            choke_escape_flash_count = 0
            choke_escape_effective_count = 0
            early_flash_count = 0
            early_flash_episode_flag = 0

            flash_reward_sum = 0.0
            flash_distance_delta_sum = 0.0
            flash_min_margin_gain_sum = 0.0
            flash_openness_gain_sum = 0.0

            flash_survive_5_success = 0
            flash_survive_5_fail = 0
            pending_flash_steps = []

            flash_eval_trigger_count = 0
            flash_execute_count = 0
            flash_skip_cooldown_count = 0
            flash_skip_no_safe_candidate_count = 0
            flash_skip_move_better_count = 0
            best_flash_better_than_move_count = 0
            best_move_better_than_flash_count = 0

            if self.logger:
                self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)
                if act >= 8:
                    flash_count += 1
                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                flash_info = _remain_info.get("flash_info", {})
                decision_info = _remain_info.get("decision_info", {})

                decision_triggered = bool(decision_info.get("flash_eval_triggered", False))
                flash_eval_trigger_count += int(decision_triggered)
                flash_execute_count += int(bool(decision_info.get("flash_execute", False)))
                if decision_triggered:
                    best_flash_better_than_move_count += int(
                        bool(decision_info.get("best_flash_better_than_move", False))
                    )
                    best_move_better_than_flash_count += int(
                        bool(decision_info.get("best_move_better_than_flash", False))
                    )

                skip_reason = str(decision_info.get("flash_skip_reason", ""))
                if skip_reason == "COOLDOWN":
                    flash_skip_cooldown_count += 1
                elif skip_reason == "NO_SAFE_CANDIDATE":
                    flash_skip_no_safe_candidate_count += 1
                elif skip_reason == "MOVE_BETTER":
                    flash_skip_move_better_count += 1

                flash_used = bool(flash_info.get("flash_used", False))
                flash_effective = bool(flash_info.get("flash_effective", False))

                danger_flash_count += int(bool(flash_info.get("flash_in_danger", False)))
                safe_flash_count += int(bool(flash_info.get("flash_in_safe", False)))
                unknown_flash_count += int(bool(flash_info.get("unknown_flash", 0)))
                danger_effective_flash_count += int(bool(flash_info.get("danger_effective_flash", False)))
                danger_ineffective_flash_count += int(bool(flash_info.get("danger_ineffective_flash", False)))
                escape_effective_flash_count += int(flash_info.get("escape_effective_flash", 0))
                non_escape_effective_flash_count += int(flash_info.get("non_escape_effective_flash", 0))
                escape_flash_count += int(flash_info.get("escape_flash", 0))
                invalid_flash_count += int(flash_info.get("invalid_flash", 0))
                flash_pre_in_threat_count += int(flash_info.get("flash_pre_in_threat", 0))
                flash_pre_in_near_threat_count += int(flash_info.get("flash_pre_in_near_threat", 0))
                flash_leave_threat_count += int(flash_info.get("flash_leave_threat", 0))
                flash_leave_danger_count += int(flash_info.get("flash_leave_danger", 0))
                post_flash_dead_end_count += int(bool(flash_info.get("post_flash_dead_end", False)))
                wall_cross_flash_count += int(flash_info.get("wall_cross_flash", 0))
                wall_cross_effective_count += int(flash_info.get("wall_cross_effective", 0))
                choke_escape_flash_count += int(flash_info.get("choke_escape_flash", 0))
                choke_escape_effective_count += int(flash_info.get("choke_escape_effective", 0))

                early_flash_value = int(flash_info.get("early_flash", 0))
                early_flash_count += early_flash_value
                if early_flash_value > 0:
                    early_flash_episode_flag = 1

                flash_reward_sum += float(flash_info.get("flash_reward", 0.0))
                flash_distance_delta_sum += float(flash_info.get("flash_distance_delta", 0.0))
                flash_min_margin_gain_sum += float(flash_info.get("flash_min_margin_gain", 0.0))
                flash_openness_gain_sum += float(flash_info.get("flash_openness_gain", 0.0))

                if flash_used:
                    if flash_effective:
                        effective_flash_count += 1
                    else:
                        ineffective_flash_count += 1
                    pending_flash_steps.append(step)

                matured_flash_steps = [flash_step for flash_step in pending_flash_steps if step - flash_step >= 5]
                flash_survive_5_success += len(matured_flash_steps)
                pending_flash_steps = [flash_step for flash_step in pending_flash_steps if step - flash_step < 5]

                total_reward += float(reward[0])

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                        flash_survive_5_fail += len(pending_flash_steps)
                    else:
                        final_reward[0] = 10.0
                        result_str = "WIN"
                        flash_survive_5_success += len(pending_flash_steps)

                    pending_flash_steps = []
                    flash_survive_5_total = flash_survive_5_success + flash_survive_5_fail
                    post_flash_survive_5_rate = flash_survive_5_success / max(flash_survive_5_total, 1)

                    flash_rate = flash_count / max(step, 1)
                    effective_flash_rate = effective_flash_count / max(flash_count, 1)
                    ineffective_flash_rate = ineffective_flash_count / max(flash_count, 1)
                    danger_flash_rate = danger_flash_count / max(flash_count, 1)
                    safe_flash_rate = safe_flash_count / max(flash_count, 1)
                    danger_effective_flash_rate = danger_effective_flash_count / max(danger_flash_count, 1)
                    danger_ineffective_flash_rate = danger_ineffective_flash_count / max(danger_flash_count, 1)
                    avg_flash_distance_delta_per_flash = flash_distance_delta_sum / max(flash_count, 1)
                    avg_flash_reward_per_flash = flash_reward_sum / max(flash_count, 1)

                    flash_execute_rate = flash_execute_count / max(flash_eval_trigger_count, 1)
                    flash_pre_in_threat_rate = flash_pre_in_threat_count / max(flash_count, 1)
                    flash_pre_in_near_threat_rate = flash_pre_in_near_threat_count / max(flash_count, 1)
                    flash_leave_threat_rate = flash_leave_threat_count / max(flash_count, 1)
                    flash_leave_danger_rate = flash_leave_danger_count / max(flash_count, 1)
                    avg_flash_distance_gain = flash_distance_delta_sum / max(flash_count, 1)
                    avg_flash_min_margin_gain = flash_min_margin_gain_sum / max(flash_count, 1)
                    avg_flash_openness_gain = flash_openness_gain_sum / max(flash_count, 1)
                    invalid_flash_rate = invalid_flash_count / max(flash_count, 1)
                    post_flash_dead_end_rate = post_flash_dead_end_count / max(flash_count, 1)
                    wall_cross_effective_rate = wall_cross_effective_count / max(wall_cross_flash_count, 1)
                    choke_escape_effective_rate = choke_escape_effective_count / max(choke_escape_flash_count, 1)

                    if self.logger:
                        self.logger.info(
                            f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                            f"result:{result_str} sim_score:{total_score:.1f} "
                            f"total_reward:{total_reward:.3f} "
                            f"flash_eval_trigger_count:{flash_eval_trigger_count} "
                            f"flash_execute_count:{flash_execute_count} "
                            f"flash_execute_rate:{flash_execute_rate:.4f} "
                            f"flash_skip_cooldown_count:{flash_skip_cooldown_count} "
                            f"flash_skip_no_safe_candidate_count:{flash_skip_no_safe_candidate_count} "
                            f"flash_skip_move_better_count:{flash_skip_move_better_count} "
                            f"best_flash_better_than_move_count:{best_flash_better_than_move_count} "
                            f"best_move_better_than_flash_count:{best_move_better_than_flash_count} "
                            f"flash_count:{flash_count} "
                            f"danger_flash_count:{danger_flash_count} "
                            f"safe_flash_count:{safe_flash_count} "
                            f"unknown_flash_count:{unknown_flash_count} "
                            f"flash_rate:{flash_rate:.4f} "
                            f"effective_flash_count:{effective_flash_count} "
                            f"ineffective_flash_count:{ineffective_flash_count} "
                            f"escape_effective_flash_count:{escape_effective_flash_count} "
                            f"non_escape_effective_flash_count:{non_escape_effective_flash_count} "
                            f"flash_pre_in_threat_count:{flash_pre_in_threat_count} "
                            f"flash_pre_in_near_threat_count:{flash_pre_in_near_threat_count} "
                            f"flash_leave_threat_count:{flash_leave_threat_count} "
                            f"flash_leave_danger_count:{flash_leave_danger_count} "
                            f"danger_effective_flash_count:{danger_effective_flash_count} "
                            f"danger_ineffective_flash_count:{danger_ineffective_flash_count} "
                            f"effective_flash_rate:{effective_flash_rate:.4f} "
                            f"ineffective_flash_rate:{ineffective_flash_rate:.4f} "
                            f"danger_flash_rate:{danger_flash_rate:.4f} "
                            f"safe_flash_rate:{safe_flash_rate:.4f} "
                            f"danger_effective_flash_rate:{danger_effective_flash_rate:.4f} "
                            f"danger_ineffective_flash_rate:{danger_ineffective_flash_rate:.4f} "
                            f"flash_pre_in_threat_rate:{flash_pre_in_threat_rate:.4f} "
                            f"flash_pre_in_near_threat_rate:{flash_pre_in_near_threat_rate:.4f} "
                            f"flash_leave_threat_rate:{flash_leave_threat_rate:.4f} "
                            f"flash_leave_danger_rate:{flash_leave_danger_rate:.4f} "
                            f"avg_flash_distance_gain:{avg_flash_distance_gain:.4f} "
                            f"avg_flash_min_margin_gain:{avg_flash_min_margin_gain:.4f} "
                            f"avg_flash_openness_gain:{avg_flash_openness_gain:.4f} "
                            f"invalid_flash_rate:{invalid_flash_rate:.4f} "
                            f"post_flash_dead_end_rate:{post_flash_dead_end_rate:.4f} "
                            f"early_flash_count:{early_flash_count} "
                            f"early_flash_episode_count:{early_flash_episode_flag} "
                            f"wall_cross_flash_count:{wall_cross_flash_count} "
                            f"wall_cross_effective_rate:{wall_cross_effective_rate:.4f} "
                            f"choke_escape_flash_count:{choke_escape_flash_count} "
                            f"choke_escape_effective_rate:{choke_escape_effective_rate:.4f} "
                            f"avg_flash_distance_delta_per_flash:{avg_flash_distance_delta_per_flash:.4f} "
                            f"avg_flash_reward_per_flash:{avg_flash_reward_per_flash:.4f} "
                            f"flash_survive_5_rate:{post_flash_survive_5_rate:.4f} "
                            f"post_flash_survive_5_rate:{post_flash_survive_5_rate:.4f}"
                        )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                    policy_bias=np.array(act_data.policy_bias, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "flash_count": flash_count,
                            "flash_rate": round(flash_rate, 4),
                            "effective_flash_count": effective_flash_count,
                            "ineffective_flash_count": ineffective_flash_count,
                            "escape_effective_flash_count": escape_effective_flash_count,
                            "non_escape_effective_flash_count": non_escape_effective_flash_count,
                            "effective_flash_rate": round(effective_flash_rate, 4),
                            "ineffective_flash_rate": round(ineffective_flash_rate, 4),
                            "danger_flash_count": danger_flash_count,
                            "safe_flash_count": safe_flash_count,
                            "danger_flash_rate": round(danger_flash_rate, 4),
                            "safe_flash_rate": round(safe_flash_rate, 4),
                            "unknown_flash_count": unknown_flash_count,
                            "danger_effective_flash_count": danger_effective_flash_count,
                            "danger_ineffective_flash_count": danger_ineffective_flash_count,
                            "danger_effective_flash_rate": round(danger_effective_flash_rate, 4),
                            "danger_ineffective_flash_rate": round(danger_ineffective_flash_rate, 4),
                            "flash_pre_in_threat_count": flash_pre_in_threat_count,
                            "flash_pre_in_near_threat_count": flash_pre_in_near_threat_count,
                            "flash_leave_threat_count": flash_leave_threat_count,
                            "flash_leave_danger_count": flash_leave_danger_count,
                            "escape_flash_count": escape_flash_count,
                            "invalid_flash_count": invalid_flash_count,
                            "flash_reward_sum": round(flash_reward_sum, 4),
                            "flash_distance_delta_sum": round(flash_distance_delta_sum, 4),
                            "avg_flash_distance_delta_per_flash": round(avg_flash_distance_delta_per_flash, 4),
                            "avg_flash_reward_per_flash": round(avg_flash_reward_per_flash, 4),
                            "flash_survive_5_success": flash_survive_5_success,
                            "flash_survive_5_fail": flash_survive_5_fail,
                            "flash_survive_5_rate": round(post_flash_survive_5_rate, 4),
                            "post_flash_survive_5_rate": round(post_flash_survive_5_rate, 4),
                            "flash_eval_trigger_count": flash_eval_trigger_count,
                            "flash_execute_count": flash_execute_count,
                            "flash_execute_rate": round(flash_execute_rate, 4),
                            "flash_skip_cooldown_count": flash_skip_cooldown_count,
                            "flash_skip_no_safe_candidate_count": flash_skip_no_safe_candidate_count,
                            "flash_skip_move_better_count": flash_skip_move_better_count,
                            "best_flash_better_than_move_count": best_flash_better_than_move_count,
                            "best_move_better_than_flash_count": best_move_better_than_flash_count,
                            "flash_pre_in_threat_rate": round(flash_pre_in_threat_rate, 4),
                            "flash_pre_in_near_threat_rate": round(flash_pre_in_near_threat_rate, 4),
                            "flash_leave_threat_rate": round(flash_leave_threat_rate, 4),
                            "flash_leave_danger_rate": round(flash_leave_danger_rate, 4),
                            "avg_flash_distance_gain": round(avg_flash_distance_gain, 4),
                            "avg_flash_min_margin_gain": round(avg_flash_min_margin_gain, 4),
                            "avg_flash_openness_gain": round(avg_flash_openness_gain, 4),
                            "invalid_flash_rate": round(invalid_flash_rate, 4),
                            "post_flash_dead_end_count": post_flash_dead_end_count,
                            "post_flash_dead_end_rate": round(post_flash_dead_end_rate, 4),
                            "early_flash_count": early_flash_count,
                            "early_flash_episode_count": early_flash_episode_flag,
                            "wall_cross_flash_count": wall_cross_flash_count,
                            "wall_cross_effective_count": wall_cross_effective_count,
                            "wall_cross_effective_rate": round(wall_cross_effective_rate, 4),
                            "choke_escape_flash_count": choke_escape_flash_count,
                            "choke_escape_effective_count": choke_escape_effective_count,
                            "choke_escape_effective_rate": round(choke_escape_effective_rate, 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
