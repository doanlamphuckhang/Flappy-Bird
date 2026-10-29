"""Uniform replay buffer (preallocated NumPy arrays) and an n-step return collector."""
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_shape, dtype=np.uint8):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not state_shape or any(not isinstance(n, int) or n <= 0 for n in state_shape):
            raise ValueError("state_shape must contain positive integers")
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.nsteps = np.ones(capacity, dtype=np.int64)  # steps between state and next_state
        self.ptr = 0
        self.size = 0

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.states, self.next_states, self.actions,
                                      self.rewards, self.dones, self.nsteps))

    def push(self, state, action, reward, next_state, done, nstep=1):
        """Store a transition; `done` must be True only for true termination, not truncation."""
        i = self.ptr
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.dones[i], self.nsteps[i] = next_state, done, nstep
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Return (states, actions, rewards, next_states, dones, nsteps) for random indices."""
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx], self.nsteps[idx])

    def __len__(self):
        return self.size


class NStepCollector:
    """Turn 1-step transitions into n-step ones.

    Each emitted transition is (s_t, a_t, sum_i gamma^i r_{t+i}, s_{t+k}, done, k) with k <= n.
    At the end of an episode the shorter tails are flushed too. Truncation keeps done=False so
    the target still bootstraps from s_{t+k}.
    """

    def __init__(self, n, gamma):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        self.n, self.gamma = n, gamma
        self.queue = deque()

    def push(self, state, action, reward, next_state, terminated, truncated=False):
        """Add one step; return the list of n-step transitions that are now complete."""
        self.queue.append((state, action, reward))
        ready = []
        if len(self.queue) == self.n:
            ready.append(self._pop(next_state, terminated))
        if terminated or truncated:
            while self.queue:
                ready.append(self._pop(next_state, terminated))
        return ready

    def _pop(self, next_state, done):
        k = len(self.queue)
        ret = sum(self.gamma ** i * r for i, (_, _, r) in enumerate(self.queue))
        state, action, _ = self.queue.popleft()
        return state, action, ret, next_state, done, k

    def clear(self):
        self.queue.clear()

    def __len__(self):
        return len(self.queue)
