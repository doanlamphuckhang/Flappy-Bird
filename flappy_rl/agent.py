"""Double DQN agent (van Hasselt et al., 2016) with dueling networks and n-step returns."""
import os
import random
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from .config import AgentConfig
from .models import build_network
from .replay import NStepCollector, ReplayBuffer
from .runtime import resolve_device


def linear_schedule(start, end, duration, t):
    """Linearly move from `start` to `end` over `duration` steps, then hold."""
    return start + min(1.0, t / duration) * (end - start)


class DQNAgent:
    def __init__(self, config=None, device="cpu"):
        self.config = config or AgentConfig()
        self.device = resolve_device(device)
        self.use_amp = self.device.type == "cuda"

        self.policy_net = build_network(self.config.mode).to(self.device)
        self.target_net = build_network(self.config.mode).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(self.config.buffer_capacity, self.config.state_shape,
                                   self.config.state_dtype)
        self.nstep = NStepCollector(self.config.n_step, self.config.gamma)
        self.total_steps = 0  # gradient updates
        self.env_steps = 0    # environment steps taken after warmup

    @property
    def epsilon(self):
        c = self.config
        return linear_schedule(c.epsilon_start, c.epsilon_min, c.exploration_steps, self.env_steps)

    def observe(self, state, action, reward, next_state, terminated, truncated=False, count=True):
        """Feed one environment step to the n-step collector and store finished transitions.

        `count=False` (warmup) leaves the exploration schedule untouched.
        """
        for transition in self.nstep.push(state, action, reward, next_state, terminated, truncated):
            self.memory.push(*transition)
        if count:
            self.env_steps += 1

    @torch.no_grad()
    def q_values(self, state):
        batch = torch.from_numpy(np.expand_dims(state, 0)).to(self.device)
        return self.policy_net(batch)[0].float().cpu().numpy()

    def select_action(self, state, evaluate=False):
        """Epsilon-greedy action; `evaluate=True` is purely greedy."""
        if not evaluate and random.random() < self.epsilon:
            return self.random_action()
        return int(self.q_values(state).argmax())

    def random_action(self):
        """Exploration action, biased towards "do nothing" (see AgentConfig.explore_flap_prob)."""
        return int(random.random() < self.config.explore_flap_prob)

    def _td_target(self, rewards, next_states, dones, nsteps=None):
        """r + gamma^k * Q_target(s', argmax_a Q_policy(s', a)), zeroed past termination."""
        best = self.policy_net(next_states).argmax(dim=-1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, best).float()
        discount = self.config.gamma if nsteps is None else self.config.gamma ** nsteps.float()
        return rewards + (1.0 - dones.float()) * discount * next_q

    def update(self):
        """One gradient step on a replay batch; returns the loss or None while the buffer fills."""
        cfg = self.config
        if len(self.memory) < cfg.batch_size:
            return None
        states, actions, rewards, next_states, dones, nsteps = (
            torch.from_numpy(a).to(self.device, non_blocking=True)
            for a in self.memory.sample(cfg.batch_size))
        actions, rewards, dones, nsteps = (t.unsqueeze(1) for t in (actions, rewards, dones, nsteps))

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            q = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                target = self._td_target(rewards, next_states, dones, nsteps)
        loss = self.loss_fn(q.float(), target.float())  # loss in fp32 for stability

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.total_steps += 1
        self._update_target()
        return loss.item()

    def _update_target(self):
        cfg = self.config
        if cfg.soft_update:
            with torch.no_grad():
                for target, policy in zip(self.target_net.parameters(), self.policy_net.parameters(),
                                          strict=True):
                    target.lerp_(policy, cfg.tau)
        elif self.total_steps % cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """Atomic checkpoint (write to a temp file, then rename). Replay memory is not stored."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({
            "config": asdict(self.config),
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "total_steps": self.total_steps,
            "env_steps": self.env_steps,
        }, tmp)
        os.replace(tmp, path)

    def load(self, path, training=True):
        """Load weights into this agent. `training=True` also restores optimizer and counters."""
        ckpt = _read_checkpoint(path, self.device)
        current = asdict(self.config)
        if training and any(current.get(k) != v for k, v in ckpt["config"].items()):
            raise ValueError("Resuming needs the saved config; use DQNAgent.from_checkpoint")
        self.policy_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
        if training:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scaler_state_dict"):
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.env_steps = ckpt.get("env_steps", self.total_steps)

    @classmethod
    def from_checkpoint(cls, path, device="cpu", training=False):
        """Rebuild an agent from the config stored in a checkpoint.

        Inference (default) allocates a 1-slot replay buffer to save memory.
        """
        device = resolve_device(device)
        config = _config_from_checkpoint(_read_checkpoint(path, device))
        if not training:
            config = config.replace(buffer_capacity=1)
        agent = cls(config, device)
        agent.load(path, training=training)
        return agent


def _read_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "mode" in ckpt:  # older checkpoints kept the mode outside the config
        ckpt["config"] = {**ckpt.get("config", {}), "mode": ckpt["mode"]}
    return ckpt


def _config_from_checkpoint(ckpt):
    known = {f.name for f in fields(AgentConfig)}
    return AgentConfig(**{k: v for k, v in ckpt["config"].items() if k in known})
