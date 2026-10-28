"""Uniform replay buffer backed by preallocated NumPy arrays."""
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
        self.ptr = 0
        self.size = 0

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.states, self.next_states, self.actions,
                                      self.rewards, self.dones))

    def push(self, state, action, reward, next_state, done):
        """Store a transition; `done` must be True only for true termination, not truncation."""
        i = self.ptr
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.dones[i] = next_state, done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size
