#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent class for the DIY PPO stage-3A agent.
"""

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs, preprocessor=None, extra_info=None):
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {
            "reward": reward,
            "flash_info": self.preprocessor.last_flash_info,
            "decision_info": self.preprocessor.last_decision_info,
        }
        remain_info.update(self.preprocessor.last_flash_info)
        remain_info.update(self.preprocessor.last_decision_info)
        return obs_data, remain_info

    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value = self._run_model(feature)
        planner_result, policy_bias, prob = self._planner_adjusted_policy(logits, legal_action)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)
        self.preprocessor.record_policy_decision(planner_result, action, d_action, policy_bias)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
                policy_bias=list(policy_bias),
            )
        ]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        logits, value = self._run_model(obs_data.feature)
        planner_result, policy_bias, prob = self._planner_adjusted_policy(logits, obs_data.legal_action)
        action = int(planner_result["chosen_action"]) if planner_result.get("flash_eval_triggered") else self._legal_sample(prob, use_max=True)
        if action < 0:
            action = self._legal_sample(prob, use_max=True)
        self.preprocessor.record_policy_decision(planner_result, action, action, policy_bias)
        act_data = ActData(
            action=[action],
            d_action=[action],
            prob=list(prob),
            value=value,
            policy_bias=list(policy_bias),
        )
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        checkpoint = torch.load(model_file_path, map_location=self.device)
        model_state = self.model.state_dict()
        compatible_state = {}
        skipped_keys = []

        for key, value in checkpoint.items():
            if key in model_state and model_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped_keys.append(key)

        model_state.update(compatible_state)
        self.model.load_state_dict(model_state)
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")
            if skipped_keys:
                self.logger.info(
                    f"skip incompatible checkpoint tensors during load: {','.join(skipped_keys)}"
                )

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]
        return logits_np, value_np

    def _planner_adjusted_policy(self, logits, legal_action):
        legal_action_np = np.array(legal_action, dtype=np.float32)
        planner_result = self.preprocessor.plan_flash_escape()
        policy_bias = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        if Config.ENABLE_FLASH_ESCAPE_V1 and planner_result is not None:
            raw_bias = planner_result.get("action_bias", [0.0] * Config.ACTION_NUM)
            policy_bias = np.array(raw_bias, dtype=np.float32)
            policy_bias += self._planner_flash_gate_bias(planner_result, legal_action_np)
        prob = self._legal_soft_max(logits + policy_bias, legal_action_np)
        return planner_result, policy_bias, prob

    def _planner_flash_gate_bias(self, planner_result, legal_action):
        gate_bias = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        flash_actions = [action for action in range(8, 16) if legal_action[action] > 0.0]
        if not flash_actions:
            return gate_bias

        flash_execute = bool(planner_result.get("flash_execute", False))
        chosen_action = int(planner_result.get("chosen_action", -1))
        chosen_flash_valid = 8 <= chosen_action < 16 and legal_action[chosen_action] > 0.0

        if not flash_execute or not chosen_flash_valid:
            for action in flash_actions:
                gate_bias[action] -= Config.FLASH_GATE_BLOCK_BIAS
            return gate_bias

        for action in flash_actions:
            if action == chosen_action:
                gate_bias[action] += Config.FLASH_GATE_CHOSEN_BIAS
            else:
                gate_bias[action] -= Config.FLASH_GATE_BLOCK_BIAS

        for action in range(8):
            if legal_action[action] > 0.0:
                gate_bias[action] -= Config.FLASH_GATE_MOVE_SUPPRESS_WHEN_EXECUTE

        return gate_bias

    def _legal_soft_max(self, input_hidden, legal_action):
        weight, eps = 1e20, 1e-5
        tmp = input_hidden - weight * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -weight, 1)
        tmp = (np.exp(tmp) + eps) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
