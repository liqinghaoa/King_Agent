#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Simple MLP Actor-Critic model for the DIY PPO stage-1 agent.
"""

import torch.nn as nn

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy_stage1"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = 128
        mid_dim = 64
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )
        self.actor_head = make_fc_layer(mid_dim, action_num)
        self.critic_head = make_fc_layer(mid_dim, value_num)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
