from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


def _mask_from_observation(observation: th.Tensor, action_count: int) -> th.Tensor:
    mask = observation[..., -action_count:] > 0.5
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if not th.all(mask.any(dim=1)):
        raise ValueError("Every observation must have at least one valid action.")
    return mask


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def _masked_distribution(self, latent_pi: th.Tensor, obs: th.Tensor):
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise TypeError("MaskedActorCriticPolicy requires a Discrete action space.")
        logits = self.action_net(latent_pi)
        mask = _mask_from_observation(obs, self.action_space.n)
        logits = logits.masked_fill(~mask, th.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._masked_distribution(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._masked_distribution(latent_pi, obs)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(observation)
        latent_pi = (
            self.mlp_extractor.forward_actor(features)
            if not self.share_features_extractor
            else self.mlp_extractor(features)[0]
        )
        return self._masked_distribution(latent_pi, observation).get_actions(
            deterministic=deterministic
        )


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
