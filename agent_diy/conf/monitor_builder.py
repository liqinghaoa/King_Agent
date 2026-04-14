#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration for the DIY PPO stage-1 agent.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("gorge_chase_diy")
        .add_group(
            group_name="algorithm",
            group_name_en="algorithm",
        )
        .add_panel(
            name="reward",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="total_loss",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="value_loss",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="policy_loss",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="entropy_loss",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        .add_group(
            group_name="episode",
            group_name_en="episode",
        )
        .add_panel(
            name="episode_steps",
            name_en="episode_steps",
            type="line",
        )
        .add_metric(
            metrics_name="episode_steps",
            expr="avg(episode_steps{})",
        )
        .end_panel()
        .add_panel(
            name="episode_cnt",
            name_en="episode_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="episode_cnt",
            expr="avg(episode_cnt{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
