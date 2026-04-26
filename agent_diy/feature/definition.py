#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Sample definitions for recurrent discrete SAC.
"""

import numpy as np

from common_python.utils.common_func import create_cls

from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, temporal_feature=None, legal_action=None)

ActData = create_cls("ActData", action=None, d_action=None, prob=None)

TransitionData = create_cls(
    "TransitionData",
    obs=None,
    temporal_obs=None,
    legal_action=None,
    act=None,
    reward=None,
    next_obs=None,
    next_temporal_obs=None,
    next_legal_action=None,
    done=None,
)

SampleData = create_cls("SampleData", npdata=Config.PACKED_SEQUENCE_DIM)


def _pad_or_stack(step_list, key, seq_len, dtype=np.float32, trailing_shape=None):
    values = []
    for step in step_list:
        value = getattr(step, key)
        values.append(np.asarray(value, dtype=dtype))

    if not values:
        if trailing_shape is None:
            return np.zeros((seq_len,), dtype=dtype)
        return np.zeros((seq_len, *trailing_shape), dtype=dtype)

    value_shape = trailing_shape if trailing_shape is not None else values[0].shape
    output = np.zeros((seq_len, *value_shape), dtype=dtype)
    for idx, value in enumerate(values):
        output[idx] = np.asarray(value, dtype=dtype).reshape(value_shape)
    return output


def _build_sequence_window(step_list):
    seq_len = Config.SEQ_LEN
    valid_len = len(step_list)
    mask_seq = np.zeros((seq_len, 1), dtype=np.float32)
    mask_seq[:valid_len, 0] = 1.0

    obs_seq = _pad_or_stack(
        step_list,
        "obs",
        seq_len,
        dtype=np.float32,
        trailing_shape=(Config.DIM_OF_OBSERVATION,),
    )
    temporal_obs_seq = _pad_or_stack(
        step_list,
        "temporal_obs",
        seq_len,
        dtype=np.float32,
        trailing_shape=(Config.DIM_OF_TEMPORAL_OBSERVATION,),
    )
    legal_action_seq = _pad_or_stack(
        step_list,
        "legal_action",
        seq_len,
        dtype=np.float32,
        trailing_shape=(Config.ACTION_NUM,),
    )
    act_seq = _pad_or_stack(
        step_list,
        "act",
        seq_len,
        dtype=np.int64,
        trailing_shape=(1,),
    ).astype(np.float32, copy=False)
    reward_seq = _pad_or_stack(
        step_list,
        "reward",
        seq_len,
        dtype=np.float32,
        trailing_shape=(1,),
    )
    next_obs_seq = _pad_or_stack(
        step_list,
        "next_obs",
        seq_len,
        dtype=np.float32,
        trailing_shape=(Config.DIM_OF_OBSERVATION,),
    )
    next_temporal_obs_seq = _pad_or_stack(
        step_list,
        "next_temporal_obs",
        seq_len,
        dtype=np.float32,
        trailing_shape=(Config.DIM_OF_TEMPORAL_OBSERVATION,),
    )
    next_legal_action_seq = _pad_or_stack(
        step_list,
        "next_legal_action",
        seq_len,
        dtype=np.float32,
        trailing_shape=(Config.ACTION_NUM,),
    )
    done_seq = _pad_or_stack(
        step_list,
        "done",
        seq_len,
        dtype=np.float32,
        trailing_shape=(1,),
    )

    packed_seq = np.concatenate(
        [
            obs_seq,
            temporal_obs_seq,
            legal_action_seq,
            act_seq,
            reward_seq,
            next_obs_seq,
            next_temporal_obs_seq,
            next_legal_action_seq,
            done_seq,
            mask_seq,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    return SampleData(npdata=packed_seq.reshape(Config.PACKED_SEQUENCE_DIM))


def sample_process(list_sample_data):
    """Slice an episode's transitions into fixed-length recurrent windows."""
    if not list_sample_data:
        return []

    seq_len = Config.SEQ_LEN
    stride = Config.LEARN_LEN
    total_steps = len(list_sample_data)
    windows = []

    start = 0
    while start < total_steps:
        end = min(start + seq_len, total_steps)
        window_steps = list_sample_data[start:end]
        if len(window_steps) > Config.BURN_IN:
            windows.append(_build_sequence_window(window_steps))
        start += stride

    if not windows and total_steps > Config.BURN_IN:
        windows.append(_build_sequence_window(list_sample_data[:seq_len]))

    return windows


def SampleData2NumpyData(g_data):
    return g_data.npdata


def NumpyData2SampleData(s_data):
    return SampleData(npdata=np.asarray(s_data, dtype=np.float32).reshape(Config.PACKED_SEQUENCE_DIM))
