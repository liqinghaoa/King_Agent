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

    # Flash escape strategy V1 switches.
    ENABLE_FLASH_ESCAPE_V1 = True
    DISABLE_LEGACY_FLASH_REWARD = True

    # Flash escape trigger thresholds.
    FLASH_THREAT_CHEB_DISTANCE = 1.0
    FLASH_NEAR_THREAT_CHEB_DISTANCE = 2.0
    FLASH_TRIGGER_DANGER_DISTANCE = 6.0
    FLASH_TRIGGER_NEAR_DANGER_DISTANCE = 10.0
    FLASH_OVERRIDE_MARGIN = 2.0
    FLASH_MIN_DISTANCE_GAIN = 1.0
    FLASH_MIN_OPENNESS_GAIN = 2.0
    FLASH_DEAD_END_THRESHOLD = 6
    FLASH_EARLY_STEP_THRESHOLD = 10
    FLASH_NEAR_TRIGGER_SCORE_MARGIN = 1.5
    FLASH_NEAR_TRIGGER_MIN_MARGIN_GAIN = 0.5
    FLASH_ESCAPE_DISTANCE_GAIN = 1.0
    FLASH_ESCAPE_MIN_MARGIN_GAIN = 0.5
    FLASH_DANGER_EFFECTIVE_DISTANCE_GAIN = 0.5
    FLASH_DANGER_EFFECTIVE_MIN_MARGIN_GAIN = 0.25
    FLASH_NON_TRIGGER_SUPPRESS = 2.5
    FLASH_GATE_BLOCK_BIAS = 12.0
    FLASH_GATE_CHOSEN_BIAS = 8.0
    FLASH_GATE_MOVE_SUPPRESS_WHEN_EXECUTE = 6.0

    # Flash planner scoring weights.
    FLASH_WALL_CROSS_BONUS = 2.0
    FLASH_CHOKE_ESCAPE_BONUS = 2.0
    FLASH_INVALID_PENALTY_WEIGHT = 6.0
    FLASH_LEAVE_THREAT_SCORE_WEIGHT = 5.0
    FLASH_DISTANCE_GAIN_WEIGHT = 1.0
    FLASH_MIN_MARGIN_WEIGHT = 1.5
    FLASH_OPENNESS_WEIGHT = 0.7

    # Planner-adjusted policy logits.
    FLASH_POLICY_BIAS_SCALE = 1.0
    FLASH_POLICY_MIN_SUPPRESS = 1.0
