#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for the discrete SAC experiment.
"""

import numpy as np


class Config:
    """Discrete SAC hyperparameters, feature dimensions, and recurrent setup."""

    HERO_FEAT_DIM = 4
    MONSTER_FEAT_DIM = 7
    MAP_FEAT_DIM = 16
    TARGET_FEAT_DIM = 6
    PATH_FEAT_DIM = 16
    LEGAL_ACTION_DIM = 16
    PROGRESS_FEAT_DIM = 10
    ACTION_NUM = 16

    FEATURES = [
        HERO_FEAT_DIM,
        MONSTER_FEAT_DIM,
        MONSTER_FEAT_DIM,
        MAP_FEAT_DIM,
        TARGET_FEAT_DIM,
        PATH_FEAT_DIM,
        LEGAL_ACTION_DIM,
        PROGRESS_FEAT_DIM,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    HERO_TEMPORAL_DIM = 9
    MONSTER_TEMPORAL_DIM = 14
    PAIR_TEMPORAL_DIM = 5
    RISK_SUMMARY_DIM = 16
    LAST_ACTION_FEAT_DIM = LEGAL_ACTION_DIM
    TEMPORAL_FEATURES = [
        HERO_TEMPORAL_DIM,
        MONSTER_TEMPORAL_DIM,
        MONSTER_TEMPORAL_DIM,
        PAIR_TEMPORAL_DIM,
        RISK_SUMMARY_DIM,
        LAST_ACTION_FEAT_DIM,
    ]
    TEMPORAL_FEATURE_SPLIT_SHAPE = TEMPORAL_FEATURES
    DIM_OF_TEMPORAL_OBSERVATION = sum(TEMPORAL_FEATURE_SPLIT_SHAPE)

    USE_RECURRENT = True
    USE_GRU = False
    SEQ_LEN = 12
    BURN_IN = 4
    LEARN_LEN = 8

    SEQUENCE_FIELD_SPLIT_SHAPE = [
        DIM_OF_OBSERVATION,
        DIM_OF_TEMPORAL_OBSERVATION,
        ACTION_NUM,
        1,
        1,
        DIM_OF_OBSERVATION,
        DIM_OF_TEMPORAL_OBSERVATION,
        ACTION_NUM,
        1,
        1,
    ]
    PACKED_STEP_DIM = sum(SEQUENCE_FIELD_SPLIT_SHAPE)
    PACKED_SEQUENCE_DIM = SEQ_LEN * PACKED_STEP_DIM

    STATIC_FEATURE_START = HERO_FEAT_DIM + 2 * MONSTER_FEAT_DIM
    STATIC_FEATURE_DIM = DIM_OF_OBSERVATION - STATIC_FEATURE_START

    HIDDEN_DIM = 128
    MID_DIM = 64

    STATIC_HIDDEN_DIM = 64
    DYNAMIC_HIDDEN_DIM = 64
    MONSTER_EMBED_DIM = 32
    HERO_EMBED_DIM = 32
    PAIR_EMBED_DIM = 16
    RISK_EMBED_DIM = 32
    ACTION_EMBED_DIM = 16
    LSTM_HIDDEN_DIM = 64
    LSTM_NUM_LAYERS = 1
    RECURRENT_DROPOUT = 0.0

    ENABLE_SOFT_FLASH_CANDIDATES = False
    SOFT_FLASH_TOPK = 2

    RNN_DEBUG_STATE = False
    RNN_STATE_LOG_FIRST_N_STEPS = 3
    RNN_STATE_LOG_INTERVAL = 50
    RNN_STATE_CHANGE_EPS = 1e-6
    TEMPORAL_SUMMARY_LOG = True
    TEMPORAL_SUMMARY_INTERVAL = 1
    PLANNER_TEMPORAL_LOG = True

    GAMMA = 0.99
    TAU = 0.005

    POLICY_LR = 3e-4
    CRITIC_LR = 3e-4
    ALPHA_LR = 3e-4
    EPS = 1e-8

    # Conservative alpha setup for the first stability-focused SAC run.
    AUTO_ALPHA = False
    FIXED_ALPHA = 0.05
    INIT_ALPHA = FIXED_ALPHA
    MIN_LOG_ALPHA = float(np.log(1e-4))
    MAX_LOG_ALPHA = float(np.log(1.0))

    # If AUTO_ALPHA is re-enabled later, prefer a lower target entropy than the
    # original near-uniform policy target to avoid another temperature blow-up.
    TARGET_ENTROPY_RATIO = 0.60
    TARGET_ENTROPY = float(np.log(ACTION_NUM) * TARGET_ENTROPY_RATIO)

    GRAD_CLIP_RANGE = 5.0
