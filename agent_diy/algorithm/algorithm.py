#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Recurrent discrete SAC algorithm implementation for Gorge Chase.
"""

import copy
import os
import time

import numpy as np
import torch

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.logger = logger
        self.monitor = monitor
        self.use_auto_alpha = bool(Config.AUTO_ALPHA)

        self.policy_optimizer = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=Config.POLICY_LR,
            eps=Config.EPS,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            lr=Config.CRITIC_LR,
            eps=Config.EPS,
        )

        self.target_entropy = Config.TARGET_ENTROPY
        self.fixed_alpha = torch.tensor(
            float(Config.FIXED_ALPHA),
            dtype=torch.float32,
            device=self.device,
        )
        self.log_alpha = None
        self.alpha_optimizer = None
        if self.use_auto_alpha:
            init_log_alpha = float(
                np.clip(np.log(Config.INIT_ALPHA), Config.MIN_LOG_ALPHA, Config.MAX_LOG_ALPHA)
            )
            self.log_alpha = torch.tensor(
                init_log_alpha,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=Config.ALPHA_LR,
                eps=Config.EPS,
            )

        self.target_q1 = copy.deepcopy(self.model.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.model.q2).to(self.device)
        self.target_q1.eval()
        self.target_q2.eval()
        for param in self.target_q1.parameters():
            param.requires_grad_(False)
        for param in self.target_q2.parameters():
            param.requires_grad_(False)

        self.last_report_monitor_time = 0.0
        self.train_step = 0
        if self.logger:
            alpha_mode = "auto" if self.use_auto_alpha else "fixed"
            self.logger.info(
                f"discrete SAC alpha mode={alpha_mode} "
                f"alpha={float(self.alpha.detach().cpu().item()):.4f} "
                f"target_entropy={self.target_entropy:.4f}"
            )

    @property
    def alpha(self):
        if self.use_auto_alpha:
            return self.log_alpha.exp().clamp(
                min=np.exp(Config.MIN_LOG_ALPHA),
                max=np.exp(Config.MAX_LOG_ALPHA),
            )
        return self.fixed_alpha

    def learn(self, list_sample_data):
        """Run one recurrent discrete SAC update on a replay batch."""
        if not list_sample_data:
            return

        (
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
        ) = self._unpack_recurrent_batch(list_sample_data)

        learn_slice = slice(Config.BURN_IN, Config.SEQ_LEN)
        burn_slice = slice(0, Config.BURN_IN)
        learn_mask = mask_seq[:, learn_slice, :]
        if float(torch.sum(learn_mask).item()) <= 0.0:
            return

        self.model.set_train_mode()
        current_alpha = self.alpha.detach()
        actor_hidden = None
        q1_hidden = None
        q2_hidden = None
        next_actor_hidden = None
        next_target_q1_hidden = None
        next_target_q2_hidden = None

        if Config.BURN_IN > 0:
            with torch.no_grad():
                _, actor_hidden = self.model.policy(
                    obs_seq[:, burn_slice, :],
                    temporal_obs_seq[:, burn_slice, :],
                    return_hidden=True,
                )
                _, q1_hidden = self.model.q1(
                    obs_seq[:, burn_slice, :],
                    temporal_obs_seq[:, burn_slice, :],
                    return_hidden=True,
                )
                _, q2_hidden = self.model.q2(
                    obs_seq[:, burn_slice, :],
                    temporal_obs_seq[:, burn_slice, :],
                    return_hidden=True,
                )
                _, next_actor_hidden = self.model.actor(
                    next_obs_seq[:, burn_slice, :],
                    next_temporal_obs_seq[:, burn_slice, :],
                    return_hidden=True,
                )
                _, next_target_q1_hidden = self.target_q1(
                    next_obs_seq[:, burn_slice, :],
                    next_temporal_obs_seq[:, burn_slice, :],
                    return_hidden=True,
                )
                _, next_target_q2_hidden = self.target_q2(
                    next_obs_seq[:, burn_slice, :],
                    next_temporal_obs_seq[:, burn_slice, :],
                    return_hidden=True,
                )

        with torch.no_grad():
            next_logits_seq = self.model.policy(
                next_obs_seq[:, learn_slice, :],
                next_temporal_obs_seq[:, learn_slice, :],
                hidden_state=next_actor_hidden,
            )
            next_prob_seq, next_log_prob_seq = self._masked_policy(
                next_logits_seq,
                next_legal_action_seq[:, learn_slice, :],
            )
            next_q1_seq = self.target_q1(
                next_obs_seq[:, learn_slice, :],
                next_temporal_obs_seq[:, learn_slice, :],
                hidden_state=next_target_q1_hidden,
            )
            next_q2_seq = self.target_q2(
                next_obs_seq[:, learn_slice, :],
                next_temporal_obs_seq[:, learn_slice, :],
                hidden_state=next_target_q2_hidden,
            )
            next_min_q_seq = torch.minimum(next_q1_seq, next_q2_seq)
            next_v_seq = (
                next_prob_seq * (next_min_q_seq - current_alpha * next_log_prob_seq)
            ).sum(dim=-1, keepdim=True)
            target_q_seq = (
                reward_seq[:, learn_slice, :]
                + (1.0 - done_seq[:, learn_slice, :]) * Config.GAMMA * next_v_seq
            )

        q1_all_seq = self.model.q1(
            obs_seq[:, learn_slice, :],
            temporal_obs_seq[:, learn_slice, :],
            hidden_state=q1_hidden,
        )
        q2_all_seq = self.model.q2(
            obs_seq[:, learn_slice, :],
            temporal_obs_seq[:, learn_slice, :],
            hidden_state=q2_hidden,
        )
        q1_seq = q1_all_seq.gather(dim=-1, index=act_seq[:, learn_slice, :])
        q2_seq = q2_all_seq.gather(dim=-1, index=act_seq[:, learn_slice, :])

        q1_loss = self._masked_mean(
            (q1_seq - target_q_seq) ** 2,
            learn_mask,
        )
        q2_loss = self._masked_mean(
            (q2_seq - target_q_seq) ** 2,
            learn_mask,
        )
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            Config.GRAD_CLIP_RANGE,
        )
        self.critic_optimizer.step()

        logits_seq = self.model.policy(
            obs_seq[:, learn_slice, :],
            temporal_obs_seq[:, learn_slice, :],
            hidden_state=actor_hidden,
        )
        prob_seq, log_prob_seq = self._masked_policy(
            logits_seq,
            legal_action_seq[:, learn_slice, :],
        )
        with torch.no_grad():
            q1_pi_seq = self.model.q1(
                obs_seq[:, learn_slice, :],
                temporal_obs_seq[:, learn_slice, :],
                hidden_state=q1_hidden,
            )
            q2_pi_seq = self.model.q2(
                obs_seq[:, learn_slice, :],
                temporal_obs_seq[:, learn_slice, :],
                hidden_state=q2_hidden,
            )
            min_q_pi_seq = torch.minimum(q1_pi_seq, q2_pi_seq)
        policy_loss_per_step = (
            prob_seq * (current_alpha * log_prob_seq - min_q_pi_seq)
        ).sum(dim=-1, keepdim=True)
        policy_loss = self._masked_mean(policy_loss_per_step, learn_mask)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), Config.GRAD_CLIP_RANGE)
        self.policy_optimizer.step()

        entropy_seq = -(prob_seq.detach() * log_prob_seq.detach()).sum(dim=-1, keepdim=True)
        entropy_learn = entropy_seq
        alpha_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.use_auto_alpha:
            alpha_gap = (self.target_entropy - entropy_learn).detach()
            alpha_loss = -self._masked_mean(self.log_alpha * alpha_gap, learn_mask)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(Config.MIN_LOG_ALPHA, Config.MAX_LOG_ALPHA)

        self._soft_update_targets()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            alpha_value = float(self.alpha.detach().cpu().item())
            reward_mean = self._masked_mean(reward_seq[:, learn_slice, :], learn_mask)
            entropy_mean = self._masked_mean(entropy_learn, learn_mask)
            results = {
                "reward": round(float(reward_mean.item()), 4),
                "total_loss": round(float((critic_loss + policy_loss + alpha_loss).item()), 4),
                "critic_loss": round(float(critic_loss.item()), 4),
                "q1_loss": round(float(q1_loss.item()), 4),
                "q2_loss": round(float(q2_loss.item()), 4),
                "policy_loss": round(float(policy_loss.item()), 4),
                "alpha_loss": round(float(alpha_loss.item()), 4),
                "alpha": round(alpha_value, 4),
                "entropy": round(float(entropy_mean.item()), 4),
            }
            if self.logger:
                self.logger.info(
                    f"[train] total_loss:{results['total_loss']} "
                    f"critic_loss:{results['critic_loss']} "
                    f"policy_loss:{results['policy_loss']} "
                    f"alpha_loss:{results['alpha_loss']} "
                    f"alpha:{results['alpha']} "
                    f"entropy:{results['entropy']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def build_checkpoint(self):
        """Collect algorithm state that should travel with the model."""
        checkpoint = {
            "model_state": self.model.state_dict(),
            "target_q1_state": self.target_q1.state_dict(),
            "target_q2_state": self.target_q2.state_dict(),
            "alpha_mode": "auto" if self.use_auto_alpha else "fixed",
        }
        if self.use_auto_alpha:
            checkpoint["log_alpha"] = float(self.log_alpha.detach().cpu().item())
        else:
            checkpoint["fixed_alpha"] = float(self.fixed_alpha.detach().cpu().item())
        return checkpoint

    def load_checkpoint(self, checkpoint):
        """Load algorithm state from a checkpoint payload."""
        if not isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
            self.target_q1.load_state_dict(self.model.q1.state_dict())
            self.target_q2.load_state_dict(self.model.q2.state_dict())
            return

        model_state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(model_state)

        target_q1_state = checkpoint.get("target_q1_state")
        target_q2_state = checkpoint.get("target_q2_state")
        if target_q1_state is None:
            self.target_q1.load_state_dict(self.model.q1.state_dict())
        else:
            self.target_q1.load_state_dict(target_q1_state)

        if target_q2_state is None:
            self.target_q2.load_state_dict(self.model.q2.state_dict())
        else:
            self.target_q2.load_state_dict(target_q2_state)

        if self.use_auto_alpha:
            log_alpha = checkpoint.get("log_alpha")
            if log_alpha is not None:
                with torch.no_grad():
                    self.log_alpha.copy_(
                        torch.tensor(
                            float(np.clip(log_alpha, Config.MIN_LOG_ALPHA, Config.MAX_LOG_ALPHA)),
                            device=self.device,
                        )
                    )
        else:
            fixed_alpha = checkpoint.get("fixed_alpha")
            with torch.no_grad():
                if fixed_alpha is not None and checkpoint.get("alpha_mode") == "fixed":
                    self.fixed_alpha.fill_(float(fixed_alpha))
                else:
                    self.fixed_alpha.fill_(float(Config.FIXED_ALPHA))

    def _unpack_recurrent_batch(self, list_sample_data):
        packed_batch = torch.as_tensor(
            np.stack(
                [
                    np.asarray(sample.npdata, dtype=np.float32).reshape(Config.PACKED_SEQUENCE_DIM)
                    for sample in list_sample_data
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, Config.SEQ_LEN, Config.PACKED_STEP_DIM)

        (
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
        ) = torch.split(packed_batch, Config.SEQUENCE_FIELD_SPLIT_SHAPE, dim=-1)

        return (
            obs_seq,
            temporal_obs_seq,
            legal_action_seq,
            act_seq.long(),
            reward_seq,
            next_obs_seq,
            next_temporal_obs_seq,
            next_legal_action_seq,
            done_seq,
            mask_seq,
        )

    def _masked_policy(self, logits, legal_action):
        legal_action = torch.where(
            legal_action.sum(dim=-1, keepdim=True) > 0,
            legal_action,
            torch.ones_like(legal_action),
        )
        masked_logits = logits.masked_fill(legal_action < 0.5, -1e9)
        prob = torch.softmax(masked_logits, dim=-1)
        prob = prob * legal_action
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_prob = torch.log(prob.clamp_min(1e-8))
        return prob, log_prob

    def _masked_mean(self, value, mask):
        weighted = value * mask
        denom = mask.sum().clamp_min(1.0)
        return weighted.sum() / denom

    def _soft_update_targets(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.model.q1.parameters()):
                target_param.data.mul_(1.0 - Config.TAU).add_(Config.TAU * param.data)
            for target_param, param in zip(self.target_q2.parameters(), self.model.q2.parameters()):
                target_param.data.mul_(1.0 - Config.TAU).add_(Config.TAU * param.data)
