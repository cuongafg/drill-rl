from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from sb3_contrib.common.maskable.policies import (
    MaskableActorCriticPolicy as BaseMaskableActorCriticPolicy,
)
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution


class SharedCnnFeatureExtractor(BaseFeaturesExtractor):
    """Extract spatial features from the flattened observed 3D map.

    The environment keeps the action mask at the end of the observation so
    each algorithm can apply its own masking logic. The extractor consumes
    the observed-value and known-location volumes with a true 3D convolution,
    plus the normalized position.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        field_shape: tuple[int, int, int] = (101, 101, 21),
        output_dim: int = 128,
    ) -> None:
        if len(observation_space.shape) != 1:
            raise ValueError("SharedCnnFeatureExtractor requires a flat observation.")
        nx, ny, nz = (int(value) for value in field_shape)
        map_size = 2 * nx * ny * nz
        expected_size = map_size + 2
        if observation_space.shape[0] < expected_size:
            raise ValueError(
                f"Observation has {observation_space.shape[0]} values but requires "
                f"at least {expected_size} for field_shape={field_shape}."
            )
        super().__init__(observation_space, int(output_dim))
        self.field_shape = (nx, ny, nz)
        self.map_size = map_size
        self.cnn = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(128 + 2, int(output_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.reshape(-1, observations.shape[-1])
        nx, ny, nz = self.field_shape
        volume = batch[:, : self.map_size].reshape(-1, 2, nx, ny, nz)
        spatial_features = self.cnn(volume)
        position = batch[:, self.map_size : self.map_size + 2]
        return self.linear(th.cat((spatial_features, position), dim=1))


class HybridFeatureExtractor(BaseFeaturesExtractor):
    """Combine 3D spatial features with an MLP over drilling-state summaries."""

    def __init__(
        self,
        observation_space: spaces.Box,
        field_shape: tuple[int, int, int] = (101, 101, 21),
        output_dim: int = 128,
        state_dim: int = 64,
    ) -> None:
        if len(observation_space.shape) != 1:
            raise ValueError("HybridFeatureExtractor requires a flat observation.")
        nx, ny, nz = (int(value) for value in field_shape)
        map_size = 2 * nx * ny * nz
        expected_size = map_size + 2
        if observation_space.shape[0] < expected_size:
            raise ValueError(
                f"Observation has {observation_space.shape[0]} values but requires "
                f"at least {expected_size} for field_shape={field_shape}."
            )
        super().__init__(observation_space, int(output_dim))
        self.field_shape = (nx, ny, nz)
        self.map_size = map_size
        self.state_input_dim = 3 * nz + 2
        self.cnn = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_input_dim, int(state_dim)),
            nn.ReLU(),
            nn.Linear(int(state_dim), int(state_dim)),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(128 + int(state_dim), int(output_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.reshape(-1, observations.shape[-1])
        nx, ny, nz = self.field_shape
        volume = batch[:, : self.map_size].reshape(-1, 2, nx, ny, nz)
        spatial_features = self.cnn(volume)

        observed_values = volume[:, 0]
        observed_mask = volume[:, 1]
        mask_count = observed_mask.sum(dim=(1, 2))
        depth_mean = (observed_values * observed_mask).sum(dim=(1, 2)) / mask_count.clamp_min(1.0)
        centered = (observed_values - depth_mean[:, None, None, :]) * observed_mask
        depth_std = th.sqrt(
            (centered.square().sum(dim=(1, 2)) / mask_count.clamp_min(1.0)).clamp_min(1e-8)
        )
        depth_fraction = observed_mask.mean(dim=(1, 2))
        state_features = th.cat(
            (
                depth_mean,
                depth_std,
                depth_fraction,
                observations[:, self.map_size : self.map_size + 2],
            ),
            dim=1,
        )
        state_features = self.state_mlp(state_features)
        return self.output(th.cat((spatial_features, state_features), dim=1))


def _mask_from_observation(observation: th.Tensor, action_count: int) -> th.Tensor:
    mask = observation[..., -action_count:] > 0.5
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if not th.all(mask.any(dim=1)):
        raise ValueError("Every observation must have at least one valid action.")
    return mask


class _MaskedActorCriticMixin:
    def _masked_distribution(
        self,
        latent_pi: th.Tensor,
        obs: th.Tensor,
        action_masks: th.Tensor | np.ndarray | None = None,
    ):
        if not isinstance(
            self.action_dist,
            (CategoricalDistribution, MaskableCategoricalDistribution),
        ):
            raise TypeError("MaskedActorCriticPolicy requires a Discrete action space.")
        logits = self.action_net(latent_pi)
        if action_masks is None:
            mask = _mask_from_observation(obs, self.action_space.n)
        else:
            mask = th.as_tensor(action_masks, device=logits.device, dtype=th.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != (logits.shape[0], self.action_space.n):
                raise ValueError("Action mask shape does not match the policy output.")
            if not th.all(mask.any(dim=1)):
                raise ValueError("Every observation must have at least one valid action.")
        logits = logits.masked_fill(~mask, th.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._masked_distribution(latent_pi, obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._masked_distribution(latent_pi, obs, action_masks)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> th.Tensor:
        features = self.extract_features(observation)
        latent_pi = (
            self.mlp_extractor.forward_actor(features)
            if not self.share_features_extractor
            else self.mlp_extractor(features)[0]
        )
        return self._masked_distribution(
            latent_pi, observation, action_masks
        ).get_actions(deterministic=deterministic)


class MaskedActorCriticPolicy(_MaskedActorCriticMixin, ActorCriticPolicy):
    """Masked actor-critic policy for A2C."""


class MaskedPPOActorCriticPolicy(
    _MaskedActorCriticMixin,
    BaseMaskableActorCriticPolicy,
):
    """Masked actor-critic policy compatible with MaskablePPO."""


class MaskedQNetwork(QNetwork):
    def forward(self, obs: th.Tensor) -> th.Tensor:
        q_values = super().forward(obs)
        mask = _mask_from_observation(obs, self.action_space.n)
        return q_values.masked_fill(~mask, th.finfo(q_values.dtype).min)


class MaskedDQNPolicy(DQNPolicy):
    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)

    @staticmethod
    def valid_actions(observation: np.ndarray, action_count: int) -> np.ndarray:
        mask = np.asarray(observation)[-action_count:] > 0.5
        actions = np.flatnonzero(mask)
        if actions.size == 0:
            raise ValueError("Observation has no valid actions.")
        return actions


class MaskedDQN(DQN):
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            observations = np.asarray(observation)
            if observations.ndim == 1:
                actions = self.policy.valid_actions(observations, self.action_space.n)
                return np.array([np.random.choice(actions)]), state
            actions = np.array(
                [
                    np.random.choice(
                        self.policy.valid_actions(obs, self.action_space.n)
                    )
                    for obs in observations
                ],
                dtype=np.int64,
            )
            return actions, state
        return super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
