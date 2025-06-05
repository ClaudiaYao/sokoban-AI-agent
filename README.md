# How to use the Repo

## Star_pusher

this folder contains an app which could make user play the sokoban game, or switch to AI assistant to play the sokoban (need a bit revisement to connect to a pre-trained model)

**star_pusher.py**<br>
The game playing app.

**starpusher_specify_map_ai_verify.py**<br>
Check if the AI agent could solve the sokoban with the provided actions.

## gym_sokoban

This is the original gym_sokoban repo (https://github.com/mpSchrader/gym-sokoban).

## human_demos

This folder contains some sokoban levels which include human solution.

## all_code_use_symbolic_obs.ipynb

This notebook contains all the steps needed to train the sokoban agent.

## all_code_with_forward_backward.ipynb

This notebook contains all the steps needed to reproduce the algorithm mentioned in this paper `Solving Sokoban with forward-backward reinforcement learning` (https://arxiv.org/abs/2105.01904). The content in this notebook is based on all_code_use_symbolic_obs.ipynb, but make adjustment for the forward-backward algorithm.

## llm_guide_sokoban_solving,ipynb

Use LLM to generate actions, and verify if those actions could solve the sokoban.
However, the success rate is not high.
