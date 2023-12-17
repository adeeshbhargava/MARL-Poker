# MARL-Poker

This work explores strategies to adapt single-player reinforcement
learning algorithms for competitive play in two-player adversarial games such as
poker. To effectively learn to play against a competitive opponent in the absence
of an experiment, we experiment with different strategies such as training against
a random policy, adversarial training and self-play. We conduct extensive exper-
imentation to test effectiveness of each strategy and summarize our insights into
training agents for optimal performance in competitive multiplayer environments.

# Tournament:

| Player 1             | Player 2             | Strategy             | Win-rate        | Reward (1000 Games) | Winner            |
|----------------------|----------------------|----------------------|-----------------|----------------------|------------------|
| DQN_baseline         | DQN_Independent      | Cross Competition    | 26%:71%:1%      | 1329.5               | DQN_Independent  |
| DQN_Self_Play_classic      | DQN_Self_Play_improved      | Cross Competition    | 40%:58%:2%      | 489                  | DQN_Self_Play_2  |
| DQN_Independent      | DQN_shared           | Cross Competition    | 55%:41%:4%      | 464.5                | DQN_Independent  |
| DQN_Self_Play_2      | DQN_Independent      | Cross Competition    | 58%:40%:2%      | 462                  | DQN_Self_Play_2  |

## Final Results
# Training:

| Player 1    | Player 2         | Strategy            | Win-rate        | Reward (1000 Games)  | Winner           |
|-------------|------------------|---------------------|-----------------|----------------------|------------------|
| DQN Agent   | Random Agent     | DQN_Baseline        | 81%:19%:0%      | 1191.5               | DQN_baseline     |
| DQN Agent   | DQN Agent        | Independent Learning| 51%:47%:2%      | 141.5                | DQN_Independent  |
| DQN Agent   | DQN Agent        | Shared Learning     | 49%:49%:2%      | 101                  | DQN_shared       |
| DQN Agent   | DQN Agent        | Self Play classic         | 50%:48%:2%      | 64.5                 | DQN_self_play_1  |
| DQN Agent   | DQN Agent        | Self Play improved        | 51%:47%:3%      | 157.5                | DQN_self_play_2  |

## DQN Agent vs Random Agent Visualistion on Texas Hold'em Poker

https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/79c8890d-6f0e-4076-b504-d6c79fad08ca

## Plots: Texas Holdem Poker

# Tournament

# DQN_baseline vs DQN_Independent

<img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/223ac7f6-20ee-4e4b-b44b-7092ed5f2369.png" width="300" /> | <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/bd863e10-7eff-4e23-a6a3-7b11b23a84aa.png" width="300" /> | <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/e8f3084b-6276-4e18-9040-f8417d647564.png" width="350" />

# DQN_Self_Play_1 vs DQN_Self_Play_2

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/dc23633d-96e6-4182-b815-e0bf8c68d9e9" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/779c929b-4c32-45b9-9713-430eea4f7e44" width="400" />
</p>

# DQN_Independent vs DQN_shared

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/a3672063-8dc9-481c-825f-9bb14fc47a10" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/2b1836c2-81de-4c5c-b29a-82146200167e" width="400" />
</p>

# DQN_Self_Play_2 vs DQN_Independent (Finals)

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/a55d08f8-22e8-4e93-a068-4523d60c0e5a" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/9c76076c-5d57-4327-947b-48115d58db6c" width="400" />
</p>

# Training

## DQN vs Random

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/081b9a60-0a97-4687-9e5b-40ff67574e9d" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/f86ab8e1-43d1-402b-99a2-aa1c2b7d3e11" width="400" />
</p>

## DQN vs DQN (Independent)

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/1164ad72-b0c5-4afb-9114-b2c5bbac407a" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/7d4118d3-3065-4360-b6dc-f60555afe4f5" width="400" />
</p>

## DQN vs DQN (Shared Policy)

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/7bf81cda-1c15-48bb-8753-ba40779dd031" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/eaf40f48-c79f-4b4c-adac-ba698541e53e" width="400" />
</p>

## DQN vs DQN (Self Play)

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/099878a8-6559-48f7-97eb-9e4e3c1caf50" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/ea9fe17e-c3bf-482a-903d-100df193f707" width="400" />
</p>

## DQN vs DQN (Improved Self Play)

<p float="left">
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/409a07e6-e75b-4939-b5b0-61fc69bdc72c" width="400" />
  <img src="https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/d1b3e293-ef13-4830-ab70-b3b3eac962de" width="400" />
</p>

## Installation:

PettingZoo[classic,butterfly]>=1.24.0 

Pillow>=9.4.0

ray[rllib]==2.7.0

SuperSuit>=3.9.0

torch>=1.13.1

tensorflow-probability>=0.19.0

