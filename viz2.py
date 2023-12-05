import argparse
import os

import cv2 
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
# from rllib_leduc_holdem import TorchMaskedActions
from Poker_ai_random_baseline_hack1 import TorchMaskedActions

from gymnasium.core import Env
from pettingzoo.classic import leduc_holdem_v4

ACTION2LABEL = {
    0 : 'CALL', 
    1 : 'RAISE', 
    2 : 'FOLD', 
    3 : 'CHECK',
}


parser = argparse.ArgumentParser(
    description="Render pretrained policy loaded from checkpoint"
)
parser.add_argument(
    "--checkpoint-path",
    help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
)
parser.add_argument(
    "--name",
    help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
)
parser.add_argument(
    "--total_rounds", default=5, type=int,
)
args = parser.parse_args()


if args.checkpoint_path is None:
    print("The following arguments are required: --checkpoint-path")
    exit(0)

checkpoint_path = os.path.expanduser(args.checkpoint_path)


alg_name = "DQN"
ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)


def env_creator():
    env = leduc_holdem_v4.env(render_mode='rgb_array')
    return env


def draw(pno, action, reward, ann):
    if action is None:
        label = f'{int(reward)}'
    else:
        label = ACTION2LABEL.get(action, 'invalid')

    text = f"P{pno} : {label}"

    if pno == 1:
        text_pos = (50, 300)
        color = (255, 255, 255)
    else:
        text_pos = (50, 700)
        color = (255, 255, 255)
    
    ann = cv2.putText(ann, text, text_pos, 0, 1, color, 2, cv2.LINE_AA)
    return ann


def draw_total(round, reward_sums, total_wins, ann):
    ann = cv2.putText(ann, f'Game : {round}', (25, 500), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    ann = cv2.putText(ann, f'TR : {reward_sums[1][-1]}, TW : {total_wins[1][-1]}', (300, 50), 0, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    ann = cv2.putText(ann, f'TR : {reward_sums[2][-1]}, TW : {total_wins[2][-1]}', (300, 975), 0, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return ann

env = env_creator()
env_name = "leduc_holdem_v4"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))


ray.init()
DQNAgent = Algorithm.from_checkpoint(checkpoint_path)

TOTAL_ROUNDS = args.total_rounds
reward_sums = {int(a.split("_")[1])+1: [0] for a in env.possible_agents}
total_wins = {int(a.split("_")[1])+1: [0] for a in env.possible_agents}
im = 0

all_renders = []


for e in tqdm(range(TOTAL_ROUNDS), desc='total rounds'):
    env.reset()

    renders = []
    results = []

    for ano, agent in enumerate(env.agent_iter()):
        pno = int(agent.split('_')[1])+1
        observation, reward, termination, truncation, info = env.last()
        obs = observation["observation"]
        if termination or truncation:
            action = None
        else:
            policy = DQNAgent.get_policy(agent)
            batch_obs = {
                "obs": {
                    "observation": np.expand_dims(observation["observation"], 0),
                    "action_mask": np.expand_dims(observation["action_mask"], 0),
                }
            }
            batched_action, state_out, info = policy.compute_actions_from_input_dict(
                batch_obs
            )
            single_action = batched_action[0]
            action = single_action
            

        env.step(action)
        render = env.render()

        results.append([pno, action, reward])
        renders.append(render)
    
    ann_renders = []
    for res, render in zip(results[:-2], renders[:-2]):
        pno, action, reward = res 
        ann_renders.append(draw_total(e, reward_sums, total_wins, draw(pno, action, reward, render.copy())))

    ann = renders[-1].copy()    
    for res in results[-2:]:
        pno, action, reward = res
        ann = draw_total(e, reward_sums, total_wins, draw(pno, action, reward, ann))
    ann_renders.append(ann)

    for res in results[-2:]:
        pno, action, reward = res 
        reward_sums[pno].append(reward_sums[pno][-1] + reward)
        if reward > 0:
            total_wins[pno].append(total_wins[pno][-1] + 1)
        else:
            total_wins[pno].append(total_wins[pno][-1])
    
    if TOTAL_ROUNDS == 5:        
        ann = renders[-1].copy()
        ann_renders.append(draw_total(e, reward_sums, total_wins, ann))
        for render in ann_renders:
            image = Image.fromarray(render.astype(np.uint8))
            all_renders.append(image)
            # image.save(f'/home/haoming/extreme_driving/Adeesh/RL/project/marl/render/{im}.png')
            # im += 1

if TOTAL_ROUNDS == 5:
    imageio.mimsave(f'/home/haoming/extreme_driving/Adeesh/RL/project/marl/render_{TOTAL_ROUNDS}/{args.name}.mp4', all_renders, fps=0.8)
    imageio.mimsave(f'/home/haoming/extreme_driving/Adeesh/RL/project/marl/render_{TOTAL_ROUNDS}/{args.name}.gif', all_renders, fps=0.8)

x = list(range(TOTAL_ROUNDS + 1))

plt.plot(x, reward_sums[1], label='Player 1')
plt.plot(x, reward_sums[2], label='Player 2')
plt.xlabel('Number of games')
plt.ylabel('Cumulative Reward')
plt.title('Reward growth over games')
plt.legend()
plt.savefig(f'/home/haoming/extreme_driving/Adeesh/RL/project/marl/render_{TOTAL_ROUNDS}/{args.name}_reward.png')

plt.clf()

plt.plot(x, total_wins[1], '--o', label='Player 1',)
plt.plot(x, total_wins[2], '--o', label='Player 2',)
plt.xlabel('Number of games')
plt.ylabel('Cumulative Wins')
plt.title('Wins growth over games')
plt.legend()
plt.savefig(f'/home/haoming/extreme_driving/Adeesh/RL/project/marl/render_{TOTAL_ROUNDS}/{args.name}_wins.png')

print ('done')

