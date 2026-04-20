#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for the DIY PPO stage-3A agent.
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
            flash_count = 0
            danger_flash_count = 0
            safe_flash_count = 0
            unknown_flash_count = 0
            danger_effective_flash_count = 0
            danger_ineffective_flash_count = 0
            escape_flash_count = 0
            invalid_flash_count = 0
            effective_flash_count = 0
            ineffective_flash_count = 0
            flash_reward_sum = 0.0
            flash_distance_delta_sum = 0.0
            flash_survive_5_success = 0
            flash_survive_5_fail = 0
            pending_flash_steps = []
            total_reward = 0.0

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
                flash_used = bool(_remain_info.get("flash_used", flash_info.get("flash_used", False)))
                flash_effective = bool(
                    _remain_info.get("flash_effective", flash_info.get("flash_effective", False))
                )
                danger_flash_count += int(
                    bool(_remain_info.get("flash_in_danger", flash_info.get("flash_in_danger", False)))
                )
                safe_flash_count += int(
                    bool(_remain_info.get("flash_in_safe", flash_info.get("flash_in_safe", False)))
                )
                unknown_flash_count += int(
                    bool(_remain_info.get("flash_in_unknown", flash_info.get("flash_in_unknown", False)))
                )
                danger_effective_flash_count += int(
                    bool(
                        _remain_info.get(
                            "danger_effective_flash",
                            flash_info.get("danger_effective_flash", False),
                        )
                    )
                )
                danger_ineffective_flash_count += int(
                    bool(
                        _remain_info.get(
                            "danger_ineffective_flash",
                            flash_info.get("danger_ineffective_flash", False),
                        )
                    )
                )
                escape_flash_count += int(flash_info.get("escape_flash", 0))
                invalid_flash_count += int(flash_info.get("invalid_flash", 0))
                flash_reward_sum += float(_remain_info.get("flash_reward", flash_info.get("flash_reward", 0.0)))
                flash_distance_delta_sum += float(
                    _remain_info.get("flash_distance_delta", flash_info.get("flash_distance_delta", 0.0))
                )

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
                    flash_survive_5_rate = flash_survive_5_success / max(flash_survive_5_total, 1)

                    if self.logger:
                        self.logger.info(
                            f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                            f"result:{result_str} sim_score:{total_score:.1f} "
                            f"total_reward:{total_reward:.3f} "
                            f"flash_count:{flash_count} "
                            f"flash_rate:{flash_count / max(step, 1):.4f} "
                            f"effective_flash_count:{effective_flash_count} "
                            f"ineffective_flash_count:{ineffective_flash_count} "
                            f"danger_flash_count:{danger_flash_count} "
                            f"safe_flash_count:{safe_flash_count} "
                            f"unknown_flash_count:{unknown_flash_count} "
                            f"danger_effective_flash_count:{danger_effective_flash_count} "
                            f"danger_ineffective_flash_count:{danger_ineffective_flash_count} "
                            f"escape_flash_count:{escape_flash_count} "
                            f"invalid_flash_count:{invalid_flash_count} "
                            f"flash_reward_sum:{flash_reward_sum:.4f} "
                            f"flash_distance_delta_sum:{flash_distance_delta_sum:.4f} "
                            f"flash_survive_5_success:{flash_survive_5_success} "
                            f"flash_survive_5_fail:{flash_survive_5_fail} "
                            f"flash_survive_5_rate:{flash_survive_5_rate:.4f}"
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
                            "flash_rate": round(flash_count / max(step, 1), 4),
                            "effective_flash_count": effective_flash_count,
                            "ineffective_flash_count": ineffective_flash_count,
                            "danger_flash_count": danger_flash_count,
                            "safe_flash_count": safe_flash_count,
                            "unknown_flash_count": unknown_flash_count,
                            "danger_effective_flash_count": danger_effective_flash_count,
                            "danger_ineffective_flash_count": danger_ineffective_flash_count,
                            "escape_flash_count": escape_flash_count,
                            "invalid_flash_count": invalid_flash_count,
                            "flash_reward_sum": round(flash_reward_sum, 4),
                            "flash_distance_delta_sum": round(flash_distance_delta_sum, 4),
                            "flash_survive_5_success": flash_survive_5_success,
                            "flash_survive_5_fail": flash_survive_5_fail,
                            "flash_survive_5_rate": round(flash_survive_5_rate, 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
