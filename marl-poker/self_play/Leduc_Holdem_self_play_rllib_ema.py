"""Uses Ray's RLlib to train agents to play Leduc Holdem.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
from gymnasium.spaces import Box, Discrete
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from copy import deepcopy
from pettingzoo.classic import leduc_holdem_v4
from ray.rllib.examples.policy.random_policy import RandomPolicy

torch, nn = try_import_torch()
UPDATE_FREQ = 20
DECAY_RATE = 0.8


class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space: Box,
        action_space: Discrete,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.max(torch.log(action_mask), torch.tensor(torch.finfo(torch.float32).min))

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.decay_rate = DECAY_RATE
        self.update_freq = UPDATE_FREQ
        self.weights_keys = None
        self.weights_cache = None

    def _update_weights(self, new_weights):
        """
        Exponential weighting of policy model weights
        """
        updated_weights = dict()
        for k in self.weights_keys:
            old_w = self.weights_cache[k]
            new_w = new_weights[k]
            updated_weights[k] = self.decay_rate * new_w + (1.0 - self.decay_rate) * old_w

        self.weights_cache = updated_weights
        print("[******] Performed weighted update!!")

        return updated_weights

    def on_train_result(self, *, algorithm, result, **kwargs):
        print("[***]", algorithm.iteration)

        if self.weights_cache is None:
            player_0 = algorithm.get_policy("player_0")
            self.weights_cache = deepcopy(player_0.get_weights())
            self.weights_keys = list(self.weights_cache.keys())

        if (algorithm.iteration > 0) and (algorithm.iteration % self.update_freq == 0):

            print("[******] Updating Player 1")

            player_0 = algorithm.get_policy("player_0")
            player_1 = algorithm.get_policy("player_1")

            player_0_state = deepcopy(player_0.get_state())
            player_0_state["weights"] = self._update_weights(player_0_state["weights"])

            player_1.set_state(player_0_state)
            algorithm.workers.sync_weights()

            print("[******] Player 1 is updated!!")

if __name__ == "__main__":
    ray.init()

    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
    # function that outputs the environment you wish to register.

    def env_creator():
        env = leduc_holdem_v4.env()
        return env

    env_name = "leduc_holdem_v4"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .callbacks(SelfPlayCallback)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={"custom_model": "pa_model"},
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space, act_space, {}),
                "player_1": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            policies_to_train=["player_0"]
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
        .framework(framework="torch")
        .exploration(
            exploration_config={
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        )
    )

    tune.run(
        alg_name,
        name="DQN",
        stop={"timesteps_total": 10000000},
        checkpoint_freq=100,
        local_dir="/home/haoming/extreme_driving/Adeesh/RL/project/marl/results/" + env_name +"_self_play_ema",
        config=config.to_dict(),
    )
