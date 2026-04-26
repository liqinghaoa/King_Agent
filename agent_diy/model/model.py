#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Multi-branch recurrent model for discrete SAC.
"""

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization."""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class RecurrentBranchNetwork(nn.Module):
    """Static MLP branch + dynamic recurrent branch + action head."""

    def __init__(self, output_dim):
        super().__init__()

        self.hero_encoder = MLPEncoder(
            Config.HERO_FEAT_DIM,
            Config.HERO_EMBED_DIM,
            Config.HERO_EMBED_DIM,
        )
        self.monster_encoder = MLPEncoder(
            Config.MONSTER_FEAT_DIM,
            Config.MONSTER_EMBED_DIM,
            Config.MONSTER_EMBED_DIM,
        )
        self.risk_encoder = MLPEncoder(
            Config.RISK_SUMMARY_DIM,
            Config.RISK_EMBED_DIM,
            Config.RISK_EMBED_DIM,
        )
        self.action_encoder = MLPEncoder(
            Config.LAST_ACTION_FEAT_DIM,
            Config.ACTION_EMBED_DIM,
            Config.ACTION_EMBED_DIM,
        )
        self.dynamic_encoder = MLPEncoder(
            Config.HERO_EMBED_DIM
            + 2 * Config.MONSTER_EMBED_DIM
            + Config.RISK_EMBED_DIM
            + Config.ACTION_EMBED_DIM,
            Config.DYNAMIC_HIDDEN_DIM,
            Config.DYNAMIC_HIDDEN_DIM,
        )
        self.static_encoder = MLPEncoder(
            Config.STATIC_FEATURE_DIM,
            Config.STATIC_HIDDEN_DIM,
            Config.STATIC_HIDDEN_DIM,
        )

        if Config.USE_RECURRENT:
            if Config.USE_GRU:
                self.recurrent = nn.GRU(
                    input_size=Config.DYNAMIC_HIDDEN_DIM,
                    hidden_size=Config.LSTM_HIDDEN_DIM,
                    num_layers=Config.LSTM_NUM_LAYERS,
                    batch_first=True,
                    dropout=Config.RECURRENT_DROPOUT if Config.LSTM_NUM_LAYERS > 1 else 0.0,
                )
            else:
                self.recurrent = nn.LSTM(
                    input_size=Config.DYNAMIC_HIDDEN_DIM,
                    hidden_size=Config.LSTM_HIDDEN_DIM,
                    num_layers=Config.LSTM_NUM_LAYERS,
                    batch_first=True,
                    dropout=Config.RECURRENT_DROPOUT if Config.LSTM_NUM_LAYERS > 1 else 0.0,
                )
        else:
            self.recurrent = None

        self.head = nn.Sequential(
            make_fc_layer(Config.STATIC_HIDDEN_DIM + Config.LSTM_HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            make_fc_layer(Config.HIDDEN_DIM, Config.MID_DIM),
            nn.ReLU(),
            make_fc_layer(Config.MID_DIM, output_dim),
        )

    def initial_state(self, batch_size, device):
        if not Config.USE_RECURRENT:
            return None
        shape = (Config.LSTM_NUM_LAYERS, batch_size, Config.LSTM_HIDDEN_DIM)
        if Config.USE_GRU:
            return torch.zeros(shape, dtype=torch.float32, device=device)
        h = torch.zeros(shape, dtype=torch.float32, device=device)
        c = torch.zeros(shape, dtype=torch.float32, device=device)
        return (h, c)

    def forward(self, obs, temporal_obs, hidden_state=None, return_hidden=False):
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)
            temporal_obs = temporal_obs.unsqueeze(1)

        batch_size, seq_len = obs.shape[0], obs.shape[1]
        static_embed = self._encode_static(obs)
        dynamic_embed = self._encode_dynamic(temporal_obs)

        if self.recurrent is None:
            recurrent_output = dynamic_embed
            next_hidden = None
        else:
            if hidden_state is None:
                hidden_state = self.initial_state(batch_size, obs.device)
            recurrent_output, next_hidden = self.recurrent(dynamic_embed, hidden_state)

        final_embed = torch.cat([static_embed, recurrent_output], dim=-1)
        logits_or_q = self.head(final_embed)

        if single_step:
            logits_or_q = logits_or_q[:, 0, :]

        if return_hidden:
            return logits_or_q, next_hidden
        return logits_or_q

    def _encode_static(self, obs):
        static_feature = obs[..., Config.STATIC_FEATURE_START :]
        return self.static_encoder(static_feature.reshape(-1, Config.STATIC_FEATURE_DIM)).reshape(
            obs.shape[0], obs.shape[1], Config.STATIC_HIDDEN_DIM
        )

    def _encode_dynamic(self, temporal_obs):
        hero, monster1, monster2, risk, last_action = torch.split(
            temporal_obs,
            Config.TEMPORAL_FEATURE_SPLIT_SHAPE,
            dim=-1,
        )
        hero_embed = self.hero_encoder(hero.reshape(-1, Config.HERO_FEAT_DIM))
        monster1_embed = self.monster_encoder(monster1.reshape(-1, Config.MONSTER_FEAT_DIM))
        monster2_embed = self.monster_encoder(monster2.reshape(-1, Config.MONSTER_FEAT_DIM))
        risk_embed = self.risk_encoder(risk.reshape(-1, Config.RISK_SUMMARY_DIM))
        action_embed = self.action_encoder(last_action.reshape(-1, Config.LAST_ACTION_FEAT_DIM))
        dynamic_input = torch.cat(
            [hero_embed, monster1_embed, monster2_embed, risk_embed, action_embed],
            dim=-1,
        )
        dynamic_embed = self.dynamic_encoder(dynamic_input)
        return dynamic_embed.reshape(
            temporal_obs.shape[0],
            temporal_obs.shape[1],
            Config.DYNAMIC_HIDDEN_DIM,
        )


class PolicyNetwork(RecurrentBranchNetwork):
    def __init__(self):
        super().__init__(Config.ACTION_NUM)


class QNetwork(RecurrentBranchNetwork):
    def __init__(self):
        super().__init__(Config.ACTION_NUM)


class Model(nn.Module):
    """Discrete SAC actor + twin recurrent Q critics."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_discrete_sac_recurrent"
        self.device = device

        self.actor = PolicyNetwork()
        self.q1 = QNetwork()
        self.q2 = QNetwork()

    def policy(self, obs, temporal_obs, hidden_state=None, return_hidden=False):
        return self.actor(
            obs,
            temporal_obs,
            hidden_state=hidden_state,
            return_hidden=return_hidden,
        )

    def q_values(self, obs, temporal_obs, hidden_state=None, return_hidden=False):
        q1 = self.q1(obs, temporal_obs, hidden_state=hidden_state, return_hidden=return_hidden)
        q2 = self.q2(obs, temporal_obs, hidden_state=hidden_state, return_hidden=return_hidden)
        return q1, q2

    def initial_actor_state(self, batch_size=1, device=None):
        device = device or self.device
        return self.actor.initial_state(batch_size, device)

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
