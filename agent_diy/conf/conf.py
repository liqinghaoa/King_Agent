#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for the DIY PPO stage-3A agent.
"""


class Config:
    # Stage 3A keeps the compact vector layout, but expands legal_action
    # from 8 movement actions to the full 16-action environment space:
    # hero_self(4), monster_1(5), monster_2(5), local_map(16),
    # legal_action(16), progress(2).
    FEATURES = [
        4,
        5,
        5,
        16,
        16,
        2,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Compatibility aliases used by the original DIY template.
    USE_CNN = False
    VIEW_SIZE = 0
    FEATURE_VECTOR_SHAPE = (DIM_OF_OBSERVATION,)
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    # Full action space: 0-7 movement, 8-15 flash.
    ACTION_NUM = 16
    ACTION_SHAPE = (ACTION_NUM,)

    VALUE_NUM = 1
    VALUE_SHAPE = (VALUE_NUM,)

    # PPO hyperparameters, aligned with agent_ppo for easier comparison.
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    START_LR = INIT_LEARNING_RATE_START
    BETA_START = 0.001
    ENTROPY_LOSS_COEFF = BETA_START
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    VALUE_LOSS_COEFF = VF_COEF
    GRAD_CLIP_RANGE = 0.5
