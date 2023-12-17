# MARL-Poker

## Installation:

PettingZoo[classic,butterfly]>=1.24.0 

Pillow>=9.4.0

ray[rllib]==2.7.0

SuperSuit>=3.9.0

torch>=1.13.1

tensorflow-probability>=0.19.0

## Final Results
# Training:

| Player 1    | Player 2       | Strategy           | Win-rate     | Reward (1000 Games) | Winner           |
|-------------|-----------------|--------------------|--------------|----------------------|------------------|
| DQN Agent   | Random Agent    | DQN_Baseline       | 81%:19%:0%   | 1191.5               | DQN_baseline     |
| DQN Agent   | DQN Agent        | Independent Learning | 51%:47%:2%  | 141.5                | DQN_Independent  |
| DQN Agent   | DQN Agent        | Shared Learning    | 49%:49%:2%   | 101                  | DQN_shared       |
| DQN Agent   | DQN Agent        | Self Play 1        | 50%:48%:2%   | 64.5                 | DQN_self_play_1  |
| DQN Agent   | DQN Agent        | Self Play 2        | 51%:47%:3%   | 157.5                | DQN_self_play_2  |

# Tournament:

| Player 1             | Player 2             | Strategy             | Win-rate        | Reward (1000 Games) | Winner            |
|----------------------|----------------------|----------------------|-----------------|----------------------|-------------------|
| DQN_baseline         | DQN_Independent      | Cross Competition    | 26%:71%:1%      | 1329.5               | DQN_Independent  |
| DQN_Self_Play_1      | DQN_Self_Play_2      | Cross Competition    | 40%:58%:2%      | 489                  | DQN_Self_Play_2  |
| DQN_Independent      | DQN_shared            | Cross Competition    | 55%:41%:4%      | 464.5                | DQN_Independent  |
| DQN_Self_Play_2      | DQN_Independent      | Cross Competition    | 58%:40%:2%      | 462                  | DQN_Self_Play_2  |


DQN Agent vs Random Agent Visualistion on Texas Hold'em Poker

https://github.com/adeeshbhargava/MARL-Poker/assets/116693172/79c8890d-6f0e-4076-b504-d6c79fad08ca

