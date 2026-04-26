#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration builder for discrete SAC.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """Create monitor panels for SAC training and episode diagnostics."""
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("Gorge Chase")
        .add_group(
            group_name="Algorithm",
            group_name_en="algorithm",
        )
        .add_panel(name="Reward", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="Total Loss", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="Critic Loss", name_en="critic_loss", type="line")
        .add_metric(metrics_name="critic_loss", expr="avg(critic_loss{})")
        .add_metric(metrics_name="q1_loss", expr="avg(q1_loss{})")
        .add_metric(metrics_name="q2_loss", expr="avg(q2_loss{})")
        .end_panel()
        .add_panel(name="Policy Loss", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .add_metric(metrics_name="alpha_loss", expr="avg(alpha_loss{})")
        .end_panel()
        .add_panel(name="Entropy", name_en="entropy", type="line")
        .add_metric(metrics_name="alpha", expr="avg(alpha{})")
        .add_metric(metrics_name="entropy", expr="avg(entropy{})")
        .end_panel()
        .end_group()
        .add_group(
            group_name="Episode",
            group_name_en="episode",
        )
        .add_panel(name="Episode Perf", name_en="episode_perf", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .add_metric(metrics_name="sim_score", expr="avg(sim_score{})")
        .add_metric(metrics_name="episode_steps", expr="avg(episode_steps{})")
        .end_panel()
        .add_panel(name="Episode Resource", name_en="episode_resource", type="line")
        .add_metric(metrics_name="treasures_collected", expr="avg(treasures_collected{})")
        .add_metric(metrics_name="collected_buff", expr="avg(collected_buff{})")
        .add_metric(metrics_name="flash_count", expr="avg(flash_count{})")
        .end_panel()
        .add_panel(name="Map Context", name_en="episode_map_context", type="line")
        .add_metric(metrics_name="map_start_open_ratio", expr="avg(map_start_open_ratio{})")
        .add_metric(
            metrics_name="map_start_visible_treasures",
            expr="avg(map_start_visible_treasures{})",
        )
        .add_metric(metrics_name="map_start_visible_buffs", expr="avg(map_start_visible_buffs{})")
        .add_metric(
            metrics_name="map_start_min_monster_dist",
            expr="avg(map_start_min_monster_dist{})",
        )
        .end_panel()
        .add_panel(name="Memory Quality", name_en="episode_memory_quality", type="line")
        .add_metric(metrics_name="revisit_rate", expr="avg(revisit_rate{})")
        .add_metric(metrics_name="dead_end_entry_rate", expr="avg(dead_end_entry_rate{})")
        .add_metric(
            metrics_name="dead_end_flash_escape_rate",
            expr="avg(dead_end_flash_escape_rate{})",
        )
        .add_metric(
            metrics_name="dead_end_backtrack_rate",
            expr="avg(dead_end_backtrack_rate{})",
        )
        .add_metric(
            metrics_name="dead_end_local_mode_rate",
            expr="avg(dead_end_local_mode_rate{})",
        )
        .add_metric(
            metrics_name="dead_end_local_commit_rate",
            expr="avg(dead_end_local_commit_rate{})",
        )
        .add_metric(
            metrics_name="dead_end_flash_follow_rate",
            expr="avg(dead_end_flash_follow_rate{})",
        )
        .add_metric(
            metrics_name="dead_end_backtrack_follow_rate",
            expr="avg(dead_end_backtrack_follow_rate{})",
        )
        .add_metric(
            metrics_name="dead_end_local_follow_rate",
            expr="avg(dead_end_local_follow_rate{})",
        )
        .add_metric(
            metrics_name="post_flash_follow_rate",
            expr="avg(post_flash_follow_rate{})",
        )
        .add_metric(
            metrics_name="post_flash_pause_rate",
            expr="avg(post_flash_pause_rate{})",
        )
        .add_metric(metrics_name="discovery_step_rate", expr="avg(discovery_step_rate{})")
        .add_metric(metrics_name="map_coverage_ratio", expr="avg(map_coverage_ratio{})")
        .add_metric(
            metrics_name="hidden_treasure_memory_rate",
            expr="avg(hidden_treasure_memory_rate{})",
        )
        .add_metric(metrics_name="frontier_available_rate", expr="avg(frontier_available_rate{})")
        .add_metric(metrics_name="frontier_follow_rate", expr="avg(frontier_follow_rate{})")
        .add_metric(metrics_name="loop_survival_mode_rate", expr="avg(loop_survival_mode_rate{})")
        .add_metric(
            metrics_name="loop_anchor_follow_rate",
            expr="avg(loop_anchor_follow_rate{})",
        )
        .end_panel()
        .add_panel(name="Action Quality", name_en="episode_action_quality", type="line")
        .add_metric(metrics_name="stalled_move_rate", expr="avg(stalled_move_rate{})")
        .add_metric(metrics_name="oscillation_alert_rate", expr="avg(oscillation_alert_rate{})")
        .end_panel()
        .add_panel(name="Flash Quality", name_en="episode_flash_quality", type="line")
        .add_metric(metrics_name="effective_flash_rate", expr="avg(effective_flash_rate{})")
        .add_metric(metrics_name="wasted_flash_rate", expr="avg(wasted_flash_rate{})")
        .add_metric(metrics_name="danger_flash_rate", expr="avg(danger_flash_rate{})")
        .add_metric(metrics_name="safe_flash_rate", expr="avg(safe_flash_rate{})")
        .add_metric(
            metrics_name="danger_effective_flash_rate",
            expr="avg(danger_effective_flash_rate{})",
        )
        .add_metric(metrics_name="flash_leave_danger_rate", expr="avg(flash_leave_danger_rate{})")
        .add_metric(metrics_name="wall_cross_effective_rate", expr="avg(wall_cross_effective_rate{})")
        .add_metric(metrics_name="flash_blocked_rate", expr="avg(flash_blocked_rate{})")
        .end_panel()
        .add_panel(name="Flash Planner", name_en="episode_flash_planner", type="line")
        .add_metric(metrics_name="flash_eval_trigger_rate", expr="avg(flash_eval_trigger_rate{})")
        .add_metric(
            metrics_name="best_flash_better_than_move_rate",
            expr="avg(best_flash_better_than_move_rate{})",
        )
        .add_metric(
            metrics_name="no_flash_move_better_rate",
            expr="avg(no_flash_move_better_rate{})",
        )
        .add_metric(metrics_name="wall_cross_flash_rate", expr="avg(wall_cross_flash_rate{})")
        .add_metric(metrics_name="post_flash_dead_end_rate", expr="avg(post_flash_dead_end_rate{})")
        .end_panel()
        .add_panel(name="Flash Gains", name_en="episode_flash_gains", type="line")
        .add_metric(metrics_name="avg_flash_distance_gain", expr="avg(avg_flash_distance_gain{})")
        .add_metric(metrics_name="avg_flash_min_margin_gain", expr="avg(avg_flash_min_margin_gain{})")
        .add_metric(metrics_name="avg_flash_openness_gain", expr="avg(avg_flash_openness_gain{})")
        .add_metric(
            metrics_name="flash_pre_in_near_threat_rate",
            expr="avg(flash_pre_in_near_threat_rate{})",
        )
        .add_metric(metrics_name="flash_leave_threat_rate", expr="avg(flash_leave_threat_rate{})")
        .end_panel()
        .add_panel(name="Win Rate", name_en="win_rate", type="line")
        .add_metric(metrics_name="win_rate", expr="avg(win_rate{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
