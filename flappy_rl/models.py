"""Dueling Q-networks (Wang et al., 2016): Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""
from torch import nn


class DuelingNet(nn.Module):
    """Shared dueling head. Subclasses define `features` and the feature width."""

    def __init__(self, features, feature_dim, hidden, num_actions):
        super().__init__()
        self.features = features
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, num_actions))

    def forward(self, x):
        feat = self.features(x).flatten(start_dim=1)
        advantage = self.advantage_stream(feat)
        return self.value_stream(feat) + advantage - advantage.mean(dim=-1, keepdim=True)


class DuelingDQNCNN(DuelingNet):
    """Nature-DQN convolutional trunk for stacked 84x84 frames (uint8 in, scaled to [0, 1])."""

    def __init__(self, in_channels=4, num_actions=2):
        trunk = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        super().__init__(trunk, feature_dim=64 * 7 * 7, hidden=512, num_actions=num_actions)

    def forward(self, x):
        return super().forward(x.float() / 255.0)


class DuelingDQNMLP(DuelingNet):
    """MLP trunk for the 12-dimensional state vector."""

    def __init__(self, in_features=12, num_actions=2):
        trunk = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        super().__init__(trunk, feature_dim=256, hidden=128, num_actions=num_actions)


def build_network(mode, num_actions=2):
    if mode == "pixel":
        return DuelingDQNCNN(num_actions=num_actions)
    if mode == "state":
        return DuelingDQNMLP(num_actions=num_actions)
    raise ValueError(f"Unsupported mode: {mode}")
