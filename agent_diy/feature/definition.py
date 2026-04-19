#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and GAE computation for the DIY PPO stage-3A agent.
"""

from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    # Full 16-action mask and old-policy probability distribution.
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    for i in range(len(list_sample_data) - 1):
        if list_sample_data[i].done[0] > 0:
            list_sample_data[i].next_value = 0.0 * list_sample_data[i].value
        else:
            list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        done_mask = 1.0 - sample.done
        delta = sample.reward + gamma * sample.next_value * done_mask - sample.value
        gae = delta + gamma * lamda * done_mask * gae
        sample.advantage = gae
        sample.reward_sum = gae + sample.value


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    return _remain_info.get("reward", [0.0])
