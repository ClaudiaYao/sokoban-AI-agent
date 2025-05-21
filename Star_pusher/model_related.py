# Randomized Action
# Modified for Deepmind

# Change to L R U D
# Enhanced Sokoban RL implementation with improved learning
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import copy
import time

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

########### Claudia: all the content from this file is copied from the latest Notebook file ###############


def create_maps():
    """Create verified solvable Sokoban maps in standard character format."""
    # DeepMind Sokoban characters:
    # '#' - Wall
    # '@' - Player
    # '+' - Player on target
    # '$' - Box
    # '*' - Box on target
    # '.' - Target
    # ' ' - Empty floor

    # Very Easy maps (single box, direct path)
    very_easy_maps = [
        """
        ########
        #      #
        #      #
        #  @$. #
        #      #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        # @$.  #
        #      #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #  .$@ #
        #      #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  .   #
        #   $  #
        #   @  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #  @   #
        #  $   #
        #  .   #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #   @  #
        #  $.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #  .$  #
        #   @  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  @   #
        #  $   #
        #  .   #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #    . #
        #    $ #
        #    @ #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # @    #
        # $    #
        # .    #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        # @$   #
        #   .  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        # .    #
        #  $@  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #    . #
        #  @$  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #   .  #
        #      #
        #   $  #
        #   @  #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  @   #
        #      #
        #  $   #
        #  .   #
        #      #
        ########
        """,
        """
        ########
        #      #
        #   .  #
        #  $   #
        #   @  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #  @   #
        # $.   #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #   .$ #
        #   @  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  .   #
        #      #
        #  $   #
        #  @   #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #   @  #
        #  .$  #
        #      #
        #      #
        ########
        """
    ]

    # Easy maps (one or two boxes with simple navigation)
    easy_maps = [
        """
        ########
        #      #
        #      #
        # @$$..#
        #      #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  ..  #
        #  $$  #
        #  @   #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .    #
        # $    #
        # @$   #
        # .    #
        #      #
        ########
        """,
        """
        ########
        #      #
        #    . #
        #   $$ #
        #   @. #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        #  .#  #
        # $@$. #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  ..  #
        #  $@  #
        #  $   #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  ..  #
        #  $@$ #
        #      #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # @    #
        # $    #
        # $    #
        # ..   #
        #      #
        ########
        """,
        """
        ########
        #      #
        #   .. #
        #   $$ #
        #   @  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  @   #
        #  $$  #
        #  ..  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  @   #
        #  $   #
        #  $.  #
        #   .  #
        #      #
        ########
        """,
        """
        ########
        #      #
        #      #
        # @$   #
        # .$.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ..   #
        # $$   #
        # @    #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  @#  #
        #  $$  #
        #  ..  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  @   #
        #  $#  #
        #  $.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  .   #
        #  .#  #
        # @$$  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  @   #
        #  $   #
        #  $   #
        #  ..  #
        #      #
        ########
        """,
        """
        ########
        #      #
        #   .  #
        # @$   #
        #  $   #
        #   .  #
        #      #
        ########
        """,
        """
        ########
        #      #
        #   .  #
        #  $@  #
        #  $   #
        #   .  #
        #      #
        ########
        """,
        """
        ########
        #      #
        #   .  #
        #   $  #
        #  @$  #
        #   .  #
        #      #
        ########
        """
    ]

    # Medium maps (more boxes and obstacles)
    medium_maps = [
        """
        ########
        # .    #
        # #    #
        # $@$. #
        # #    #
        # .    #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .$@  #
        # #$$  #
        # #..  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .$.  #
        # $@$  #
        # .$.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ...  #
        # $$$  #
        #  @   #
        #  #   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # @#   #
        # $$#  #
        # ..   #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  #   #
        # @$$. #
        # .$.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # .#.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ..   #
        # #$   #
        # @$$  #
        # #..  #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  .#  #
        # $$@  #
        # .#$. #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#   #
        # $$.  #
        # @#$  #
        # .    #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  #.  #
        # $$@  #
        # .#$  #
        #  .   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # $#.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .    #
        # $$#  #
        # .@$. #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $$.  #
        # @#$  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        #  ... #
        #  $$$ #
        #  @   #
        #  #   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # #.   #
        # $@$  #
        # #$   #
        # ..   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#   #
        # $@$. #
        # $#   #
        # .    #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ...  #
        # #$$  #
        # @$   #
        # #    #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # .$.  #
        #      #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ..   #
        # $@   #
        # $$   #
        # ..   #
        #      #
        ########
        """
    ]

    # Hard maps (challenging but solvable)
    hard_maps = [
        """
        ########
        #      #
        # .**  #
        # @$   #
        # #    #
        # #    #
        #      #
        ########
        """,
        """
        ########
        #   @  #
        #  *$. #
        # #$#  #
        # .    #
        #      #
        #      #
        ########
        """,
        """
        ########
        #  @   #
        # $$   #
        # #.   #
        # #$   #
        # #..  #
        #      #
        ########
        """,
        """
        ########
        # @    #
        # $    #
        # $$   #
        # #.   #
        # #..  #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # $#.  #
        # .    #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .*.  #
        # $@$  #
        # #$#  #
        # ...  #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ...  #
        # $$$  #
        # @$   #
        # #..  #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .*   #
        # #$#  #
        # .$@  #
        # *$   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .*#  #
        # $$.  #
        # @#$  #
        # ..   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ..#  #
        # @$$  #
        # #$.  #
        # .#$  #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $$$  #
        # .@.  #
        # #$#  #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ..#  #
        # #$$  #
        # @$.  #
        # #$.  #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .*#  #
        # $@$  #
        # *$#  #
        # .    #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ...  #
        # #$$  #
        # @$#  #
        # .$   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ..#  #
        # $$@  #
        # #$.  #
        # .$   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $$@  #
        # .$$  #
        # .#   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ...  #
        # #$#  #
        # $@$  #
        # #$.  #
        #      #
        ########
        """,
        """
        ########
        #      #
        # ###  #
        # @$$. #
        # $... #
        # ##   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # $$.  #
        # .#   #
        #      #
        ########
        """,
        """
        ########
        #      #
        # .*   #
        # $@#  #
        # *$   #
        # .#   #
        #      #
        ########
        """
    ]

    # Challenging evaluation maps - specifically designed to require multiple steps
    eval_challenge_maps = [
        # Challenge 1: Requires careful navigation around obstacles
        """
        ########
        #      #
        #  ##  #
        # #@.# #
        # $  $ #
        # #..# #
        #      #
        ########
        """,

        # Challenge 2: Box extraction puzzle
        """
        ########
        #      #
        # @##  #
        # # .# #
        # $ $  #
        # #. # #
        #   #  #
        ########
        """,

        # Challenge 3: Multiple boxes with limited maneuvering space
        """
        ########
        #      #
        # .# . #
        # #@$# #
        # $ $  #
        # .# . #
        #      #
        ########
        """,

        # Challenge 4: Strategic planning required
        """
        ########
        #      #
        # .#   #
        # # $  #
        # @$$. #
        # .#   #
        #      #
        ########
        """,

        # Challenge 5: Complex multi-step solution
        """
        ########
        #      #
        #  ..  #
        # #$$# #
        # @$.  #
        #  ##  #
        #      #
        ########
        """,

        # Challenge 6: Corridor navigation
        """
        ########
        #      #
        # .##  #
        # $@$  #
        # #$.  #
        # .#   #
        #      #
        ########
        """,

        # Challenge 7: Tight spaces
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # #$$  #
        # ..   #
        #      #
        ########
        """,

        # Challenge 8: Box extraction with limited space
        """
        ########
        #      #
        # ###  #
        # @$.  #
        # $$#  #
        # ..   #
        #      #
        ########
        """,

        # Challenge 9: Tactical pushing sequence
        """
        ########
        #      #
        # .##  #
        # $@#  #
        # $.   #
        # .$   #
        #      #
        ########
        """,

        # Challenge 10: Sequential box placement
        """
        ########
        #      #
        # ...  #
        # #@#  #
        # $$$  #
        #  #   #
        #      #
        ########
        """,

        # Challenge 11: Box repositioning
        """
        ########
        #      #
        # ...  #
        # #$#  #
        # @$$  #
        # #    #
        #      #
        ########
        """,

        # Challenge 12: Careful planning required
        """
        ########
        #      #
        # ...  #
        # $$@  #
        # #$   #
        # #.   #
        #      #
        ########
        """,

        # Challenge 13: Sequential maneuvering
        """
        ########
        #      #
        # ..#  #
        # @$$  #
        # #$.  #
        #  #   #
        #      #
        ########
        """,

        # Challenge 14: Box rearrangement
        """
        ########
        #      #
        # ...  #
        # @$$  #
        # #$   #
        #  #   #
        #      #
        ########
        """,

        # Challenge 15: Strategic box placement
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # .$$  #
        #  #   #
        #      #
        ########
        """,

        # Challenge 16: Corridor navigation with multiple boxes
        """
        ########
        #      #
        # .# . #
        # $@$  #
        # $ #  #
        # . #  #
        #      #
        ########
        """,

        # Challenge 17: Confined space maneuvering
        """
        ########
        #      #
        #  #.  #
        # #$@  #
        # $.   #
        # .$   #
        #      #
        ########
        """,

        # Challenge 18: Complex multi-step planning
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # #$#  #
        # . .  #
        #      #
        ########
        """,

        # Challenge 19: Box sequencing puzzle
        """
        ########
        #      #
        # ...  #
        # #$#  #
        # @$$  #
        #  #   #
        #      #
        ########
        """,

        # Challenge 20: Strategic repositioning
        """
        ########
        #      #
        # .#.  #
        # $@$  #
        # .$$  #
        #  #.  #
        #      #
        ########
        """
    ]

    # Combine all maps into a dictionary
    char_maps = {
        'very_easy': very_easy_maps,
        'easy': easy_maps,
        'medium': medium_maps,
        'hard': hard_maps,
        'eval_challenge': eval_challenge_maps
    }

    # Convert character maps to numerical arrays
    maps = {}
    for difficulty, char_map_list in char_maps.items():
        num_maps = []
        for char_map in char_map_list:
            num_map = chars_to_numerical(char_map)
            num_maps.append(num_map)
        maps[difficulty] = num_maps

    # Combine maps for train and eval
    maps['train'] = maps['very_easy'] + maps['easy'] + maps['medium'] + maps['hard']
    # Use challenging maps for evaluation instead of just the first map of each difficulty
    maps['eval'] = maps['eval_challenge']

    return maps



# 2. Enhanced Sokoban Environment with better rewards
class SokobanEnv(gym.Env):
    """Enhanced Sokoban environment with improved rewards and curriculum learning."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, maps_type='train', render_mode="rgb_array", difficulty='curriculum'):
        super().__init__()
        self.maps = create_maps()[maps_type]
        self.render_mode = render_mode
        self.difficulty = difficulty  # Store the difficulty parameter

        # Curriculum learning parameters
        self.curriculum_phases = 5  # 0=very_easy, 1=easy, 2=easy_medium, 3=medium, 4=hard
        self.curriculum_phase = 0
        self.curriculum_threshold = 0.6  # Reduced threshold for advancement
        self.curriculum_regression_threshold = 0.25  # New separate threshold for regression
        self.curriculum_window = 30      # Increased window size
        self.success_buffer = []  # Track recent episode successes/failures

        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Sample map for dimensions
        sample_map = self.maps[0]
        self.height, self.width = sample_map.shape

        # DeepMind-style multi-channel observation space (7 channels)
        # Channel 0: Walls
        # Channel 1: Player (not on a target)
        # Channel 2: Player on a target
        # Channel 3: Box (not on a target)
        # Channel 4: Box on a target
        # Channel 5: Target (without box or player)
        # Channel 6: Empty space
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(7, self.height, self.width),  # 7 channels, DeepMind-style (CHW format)
            dtype=np.uint8
        )

        # For tracking last action (for text representation)
        self.last_direction = 0

        # For distance-based rewards
        self.prev_distances = None

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if len(self.success_buffer) >= self.curriculum_window:
            # Calculate recent success rate
            success_rate = sum(self.success_buffer) / len(self.success_buffer)

            # Consider advancing curriculum if doing well
            if success_rate >= self.curriculum_threshold:
                if self.curriculum_phase < self.curriculum_phases - 1:
                    self.curriculum_phase += 1
                    print(f"Advancing curriculum to phase {self.curriculum_phase}")
                    self.success_buffer = []  # Reset buffer
            # Consider reverting to easier level if struggling badly
            elif success_rate < self.curriculum_regression_threshold and self.curriculum_phase > 0:
                self.curriculum_phase -= 1
                print(f"Reverting to easier curriculum phase {self.curriculum_phase}")
                self.success_buffer = []  # Reset buffer

            # Reset buffer if it gets too long
            if len(self.success_buffer) > self.curriculum_window * 2:
                self.success_buffer = self.success_buffer[-self.curriculum_window:]

        # Select appropriate maps based on curriculum phase
        phase_map_ranges = [
            (0, 1),           # very_easy: index 0
            (1, 3),           # easy: indices 1-2
            (3, 5),           # easy_medium: indices 3-4
            (5, 7),           # medium: indices 5-6
            (7, len(self.maps))  # hard: indices 7+
        ]

        # Make sure curriculum_phase is within valid range
        valid_phase = max(0, min(self.curriculum_phase, len(phase_map_ranges) - 1))
        start_idx, end_idx = phase_map_ranges[valid_phase]

        # Ensure there's at least one map to choose from
        if start_idx >= end_idx:
            end_idx = start_idx + 1

        # Make sure end_idx doesn't exceed the map list length
        end_idx = min(end_idx, len(self.maps))

        map_idx = np.random.randint(start_idx, end_idx)

        # Deep copy the map
        self.room_state = np.copy(self.maps[map_idx])
        self.current_map = map_idx

        # Find player position
        player_pos = np.argwhere(self.room_state == 5)
        self.player_position = tuple(player_pos[0])

        # Find box positions
        self.box_positions = []
        for pos in np.argwhere((self.room_state == 2) | (self.room_state == 4)):
            self.box_positions.append(tuple(pos))

        # Find target positions
        self.target_positions = []
        for pos in np.argwhere((self.room_state == 3) | (self.room_state == 4)):
            self.target_positions.append(tuple(pos))

        # Count boxes on targets
        self.boxes_on_target = np.sum(self.room_state == 4)

        # Calculate distances from boxes to targets for reward shaping
        self.prev_distances = self._calculate_distances()

        # Reset step counter
        self.steps = 0

        # Create observation
        observation = self._get_observation()

        info = {
            'map_idx': map_idx,
            'boxes_on_target': self.boxes_on_target,
            'total_boxes': len(self.box_positions),
            'curriculum_phase': valid_phase  # Use the validated phase
        }

        return observation, info

    def _calculate_distances(self):
        """Calculate the sum of Manhattan distances from each box to its nearest target."""
        total_distances = 0

        # For each box, find the Manhattan distance to the closest target
        for box_pos in self.box_positions:
            if box_pos in self.target_positions:
                # Box already on target
                continue

            # Find closest target
            min_dist = float('inf')
            for target_pos in self.target_positions:
                # Skip targets that already have boxes
                if self.room_state[target_pos] == 4:
                    continue

                # Calculate Manhattan distance
                dist = abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])
                min_dist = min(min_dist, dist)

            # Add to total
            if min_dist != float('inf'):
                total_distances += min_dist

        return total_distances
    def _calculate_box_target_distances(self, state):
        """Calculate the sum of Manhattan distances from each box to its nearest target."""
        total_distances = 0

        # Find positions of boxes and targets
        box_positions = []
        for pos in np.argwhere(state == 2):  # Only non-target boxes
            box_positions.append(tuple(pos))

        target_positions = []
        for pos in np.argwhere(state == 3):  # Only empty targets
            target_positions.append(tuple(pos))

        # For each box, find the Manhattan distance to the closest target
        for box_pos in box_positions:
            # Skip if no targets left
            if len(target_positions) == 0:
                continue

            # Find closest target
            min_dist = float('inf')
            for target_pos in target_positions:
                # Calculate Manhattan distance
                dist = abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])
                min_dist = min(min_dist, dist)

            # Add to total
            if min_dist != float('inf'):
                total_distances += min_dist

        return total_distances

    def _calculate_player_positioning_reward(self):
        """Calculate reward based on player's strategic position relative to boxes and targets."""
        reward = 0

        # For each box not on target
        for box_pos in self.box_positions:
            if self.room_state[box_pos] == 4:  # Skip boxes on targets
                continue

            # Find the closest target
            closest_target = None
            min_dist = float('inf')
            for target_pos in self.target_positions:
                if self.room_state[target_pos] == 4:  # Skip targets with boxes
                    continue

                dist = abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_target = target_pos

            if closest_target is None:
                continue

            # Calculate ideal position vector (where player should be to push box toward target)
            dx = box_pos[0] - closest_target[0]
            dy = box_pos[1] - closest_target[1]

            # Determine the best side to push from
            best_push_pos = None
            if abs(dx) > abs(dy):  # Box and target differ more in x-direction
                if dx > 0:  # Target is to the left of box
                    best_push_pos = (box_pos[0] + 1, box_pos[1])  # Player should be right of box
                else:
                    best_push_pos = (box_pos[0] - 1, box_pos[1])  # Player should be left of box
            else:  # Box and target differ more in y-direction
                if dy > 0:  # Target is above the box
                    best_push_pos = (box_pos[0], box_pos[1] + 1)  # Player should be below box
                else:
                    best_push_pos = (box_pos[0], box_pos[1] - 1)  # Player should be above box

            # Check if position is valid and within bounds
            if (0 <= best_push_pos[0] < self.room_state.shape[0] and
                0 <= best_push_pos[1] < self.room_state.shape[1] and
                self.room_state[best_push_pos] not in [1, 2, 4]):  # Not wall or box

                # Reward the agent for moving toward the best pushing position
                player_dist_to_best_pos = abs(self.player_position[0] - best_push_pos[0]) + abs(self.player_position[1] - best_push_pos[1])

                # Give higher reward for being closer to the best position
                if player_dist_to_best_pos < 3:  # Close to best position
                    reward += 0.1 * (3 - player_dist_to_best_pos)

        return reward
    # Add to your SokobanEnv class
    def _is_box_accessible(self, box_pos):
        """Check if at least 2 sides of a box are accessible to the player."""
        accessible_sides = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            side_pos = (box_pos[0] + dx, box_pos[1] + dy)
            if (0 <= side_pos[0] < self.room_state.shape[0] and
                0 <= side_pos[1] < self.room_state.shape[1] and
                self.room_state[side_pos] in [0, 3, 5]):  # Empty, target or player
                accessible_sides += 1

        return accessible_sides >= 2  # Need at least 2 sides accessible

    def step(self, action):
        # Store previous state for reward calculation
        prev_state = np.copy(self.room_state)

        # Map actions to directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction = directions[action]

        # Inside step method, update this line:
        self.last_direction = int(action) if isinstance(action, np.ndarray) else action

        self.steps += 1

        # Calculate new position
        new_pos = (
            self.player_position[0] + direction[0],
            self.player_position[1] + direction[1]
        )

        # Check if new position is out of bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.room_state.shape[0] or
            new_pos[1] < 0 or new_pos[1] >= self.room_state.shape[1]):
            # Position is out of bounds, don't move
            reward = -0.1  # Penalty
            terminated = False
            truncated = False
            observation = self._get_observation()
            info = {
                'boxes_on_target': self.boxes_on_target,
                'total_boxes': len(self.box_positions),
                'all_boxes_on_target': self.boxes_on_target == len(self.box_positions),
                'steps': self.steps,
                'curriculum_phase': self.curriculum_phase
            }
            return observation, reward, terminated, truncated, info

        # Default values
        reward = -0.01  # Small penalty for steps
        terminated = False
        truncated = False
        box_moved = False
        box_pos = None
        box_new_pos = None
        box_on_target_change = 0

        # Check new position
        new_pos_value = self.room_state[new_pos]

        if new_pos_value == 1:  # Wall
            reward = -0.1  # Penalty
        else:
            if new_pos_value in [2, 4]:  # Box or box on target
                # Calculate box's new position
                box_new_pos = (
                    new_pos[0] + direction[0],
                    new_pos[1] + direction[1]
                )

                # Check if box's new position is out of bounds
                if (box_new_pos[0] < 0 or box_new_pos[0] >= self.room_state.shape[0] or
                    box_new_pos[1] < 0 or box_new_pos[1] >= self.room_state.shape[1]):
                    # Box's new position is out of bounds, don't move
                    reward = -0.2
                else:
                    box_new_pos_value = self.room_state[box_new_pos]

                    if box_new_pos_value in [0, 3]:  # Empty or target
                        # Record the original box position for reward calculation
                        box_pos = new_pos

                        # Move box
                        if new_pos_value == 2:  # Box
                            self.room_state[new_pos] = 0
                        else:  # Box on target
                            self.room_state[new_pos] = 3
                            box_on_target_change -= 1
                            self.boxes_on_target -= 1

                        # Place box
                        if box_new_pos_value == 0:  # Empty
                            self.room_state[box_new_pos] = 2
                        else:  # Target
                            self.room_state[box_new_pos] = 4
                            box_on_target_change += 1
                            self.boxes_on_target += 1

                        # Update box positions
                        self.box_positions.remove(new_pos)
                        self.box_positions.append(box_new_pos)
                        box_moved = True

                        # Move player
                        self._move_player(new_pos)
                    else:
                        reward = -0.2  # Penalty for pushing box against wall/another box
            else:
                # Move player to empty space or target
                self._move_player(new_pos)

        # Enhanced rewards

        # 1. Reward for putting boxes on targets
        if box_on_target_change > 0:
            reward += 1.5 * box_on_target_change  # Increased from 1.0
        elif box_on_target_change < 0:
            reward -= 0.5 * abs(box_on_target_change)

        # 2. Distance-based reward shaping
        if box_moved:
            current_distances = self._calculate_distances()
            distance_improvement = self.prev_distances - current_distances
            self.prev_distances = current_distances

            # Reward for moving boxes closer to targets
            if distance_improvement > 0:
                reward += 0.3 * distance_improvement  # Increased from 0.2
            # Small penalty for moving boxes away from targets
            elif distance_improvement < 0:
                reward -= 0.1 * abs(distance_improvement)

            # Additional reward for specific box movement
            if box_pos is not None and box_new_pos is not None:
                # Find closest target for this specific box
                min_dist_before = float('inf')
                min_dist_after = float('inf')

                for target_pos in self.target_positions:
                    if self.room_state[target_pos] == 4:  # Skip targets with boxes
                        continue

                    dist_before = abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])
                    dist_after = abs(box_new_pos[0] - target_pos[0]) + abs(box_new_pos[1] - target_pos[1])

                    min_dist_before = min(min_dist_before, dist_before)
                    min_dist_after = min(min_dist_after, dist_after)

                if min_dist_before != float('inf') and min_dist_after != float('inf'):
                    specific_improvement = min_dist_before - min_dist_after
                    if specific_improvement > 0:
                        reward += 0.2 * specific_improvement

        # 3. Reward for player positions near boxes
        player_pos = self.player_position
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nearby_pos = (player_pos[0] + dx, player_pos[1] + dy)
            if (0 <= nearby_pos[0] < self.room_state.shape[0] and
                0 <= nearby_pos[1] < self.room_state.shape[1]):
                if self.room_state[nearby_pos] in [2, 4]:  # Box or box on target
                    reward += 0.05  # Small reward for being near a box
                    break

        # 4. Add positioning reward to encourage going around obstacles
        try:
            positioning_reward = self._calculate_player_positioning_reward()
            reward += positioning_reward
        except IndexError:
            # If there's an index error in the positioning reward calculation, skip it
            pass

        # 5. Box accessibility and corner checks
        if box_moved:
            # Check if the box is still accessible
            try:
                if not self._is_box_accessible(box_new_pos) and self.room_state[box_new_pos] != 4:  # Not on target
                    reward -= 0.8  # Stronger penalty for making box less accessible
            except IndexError:
                # If there's an index error, skip this check
                pass

            # Check for corner/deadlock situations (simplified)
            for box_pos in self.box_positions:
                # Skip boxes on targets
                try:
                    if self.room_state[box_pos] == 4:
                        continue
                except IndexError:
                    continue

                # Check if box is in a corner
                is_corner = False
                for d1, d2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    pos1 = (box_pos[0] + d1, box_pos[1] + d2)
                    pos2 = (box_pos[0] + d2, box_pos[1] + d1)

                    if (0 <= pos1[0] < self.room_state.shape[0] and
                        0 <= pos1[1] < self.room_state.shape[1] and
                        0 <= pos2[0] < self.room_state.shape[0] and
                        0 <= pos2[1] < self.room_state.shape[1]):

                        try:
                            if self.room_state[pos1] == 1 and self.room_state[pos2] == 1:
                                is_corner = True
                                break
                        except IndexError:
                            continue

                if is_corner and box_pos not in self.target_positions:
                    reward -= 0.4  # Reduced penalty for pushing box into corner (from 0.5)

        # Check win condition
        if self.boxes_on_target == len(self.box_positions):
            reward = 10.0
            terminated = True
            self.success_buffer.append(1)  # Record success for curriculum

        # Max steps - adaptive based on curriculum phase
        max_steps = 100
        if self.curriculum_phase in [2, 3]:  # easy-medium or medium
            max_steps = 150
        elif self.curriculum_phase == 4:  # hard
            max_steps = 200

        if self.steps >= max_steps:
            truncated = True
            self.success_buffer.append(0)  # Record failure for curriculum

        observation = self._get_observation()

        info = {
            'boxes_on_target': self.boxes_on_target,
            'total_boxes': len(self.box_positions),
            'all_boxes_on_target': self.boxes_on_target == len(self.box_positions),
            'steps': self.steps,
            'curriculum_phase': self.curriculum_phase
        }

        return observation, reward, terminated, truncated, info

    def _move_player(self, new_pos):
        # Update player position state
        if self.room_state[self.player_position] == 5:
            self.room_state[self.player_position] = 0
        else:  # Player on target
            self.room_state[self.player_position] = 3

        # Place player at new position
        if self.room_state[new_pos] == 0:  # Empty
            self.room_state[new_pos] = 5
        else:  # Target
            self.room_state[new_pos] = 5

        self.player_position = new_pos

    def _get_observation(self):
        """Generate DeepMind-style observation with channels for each element type."""
        # Create empty 7-channel observation
        obs = np.zeros((7, self.height, self.width), dtype=np.uint8)

        # Fill each channel based on element type
        for i in range(self.height):
            for j in range(self.width):
                cell = self.room_state[i, j]

                if cell == 1:  # Wall
                    obs[0, i, j] = 1
                elif cell == 5:  # Player
                    if (i, j) in self.target_positions:  # Player on target
                        obs[2, i, j] = 1
                    else:  # Player not on target
                        obs[1, i, j] = 1
                elif cell == 2:  # Box not on target
                    obs[3, i, j] = 1
                elif cell == 4:  # Box on target
                    obs[4, i, j] = 1
                elif cell == 3:  # Target (without box or player)
                    obs[5, i, j] = 1
                elif cell == 0:  # Empty space
                    obs[6, i, j] = 1

        return obs

    def get_text_representation(self):
        """Returns a text-based representation of the environment using standard Sokoban characters."""
        # Map direction numbers to letters for display
        direction_map = {0: 'U', 1: 'D', 2: 'L', 3: 'R', None: 'X'}

        # Get current direction letter
        if hasattr(self, 'last_direction') and self.last_direction is not None:
            direction_key = int(self.last_direction) if isinstance(self.last_direction, np.ndarray) else self.last_direction
            direction_letter = direction_map.get(direction_key, 'X')
        else:
            direction_letter = 'X'

        # Find positions where player is on a target
        player_on_target = []
        for pos in self.target_positions:
            if self.room_state[pos] == 5:  # Player
                player_on_target.append(pos)

        # Convert to character representation
        board_text = numerical_to_chars(self.room_state, player_on_target)

        return {
            'board': board_text,
            'last_move': direction_letter
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        # For text-based rendering, use get_text_representation
        text_representation = self.get_text_representation()

        # Print the text representation to console
        print(text_representation['board'])
        print(f"Last move: {text_representation['last_move']}")

        # If you still want RGB output for visualization, you can keep this
        rgb_obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Map elements to colors
        colors = {
            0: (255, 255, 255),  # Empty (white)
            1: (0, 0, 0),        # Wall (black)
            2: (165, 42, 42),    # Box (brown)
            3: (0, 255, 0),      # Target (green)
            4: (0, 0, 255),      # Box on target (blue)
            5: (255, 0, 0)       # Player (red)
        }

        for h in range(self.height):
            for w in range(self.width):
                rgb_obs[h, w] = colors[self.room_state[h, w]]

        return rgb_obs  # Return RGB observation for visualization



# 3. Improved CNN Feature Extractor
class EnhancedSokobanCNN(BaseFeaturesExtractor):
    """CNN feature extractor for DeepMind-style Sokoban observations."""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Get dimensions (channels, height, width)
        n_input_channels = observation_space.shape[0]

        # Create CNN with appropriate padding
        self.cnn = nn.Sequential(
            # Initial convolution
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # First residual block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Second residual block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Final convolution
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + observation_space.shape)
            n_flatten = self.cnn(dummy_input).shape[1]

        # Larger FC layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# 4. Enhanced Training Callback
class EnhancedTrainingCallback(BaseCallback):
    """Enhanced callback with more detailed tracking."""
    def __init__(self, check_freq=1000, save_path="models/", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float('inf')

        # Enhanced metrics
        self.episode_rewards = [0.0]
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_count = 0
        self.step_count = 0

        # For tracking curriculum progress
        self.curriculum_phases = []

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        self.step_count += 1

        # Track episodes
        for i in range(len(self.locals['dones'])):
            if self.locals['dones'][i]:
                self.episode_count += 1

                # Store episode metrics
                self.episode_rewards.append(self.locals['rewards'][i])

                # Get info from the environment
                info = self.locals['infos'][i]
                self.episode_lengths.append(info.get('steps', 0))
                self.episode_successes.append(info.get('all_boxes_on_target', False))

                # Track curriculum phase if available
                curr_phase = info.get('curriculum_phase', -1)
                if curr_phase != -1:
                    self.curriculum_phases.append(curr_phase)

                if self.verbose > 0 and self.episode_count % 10 == 0:
                    success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)
                    print(f"Episode {self.episode_count}: reward={self.episode_rewards[-1]:.2f}, success_rate={success_rate:.2f}")

                    # Print curriculum phase if applicable
                    if len(self.curriculum_phases) > 0:
                        curr_phase = self.curriculum_phases[-1]
                        # Fix: Map curriculum_phase to appropriate name
                        phase_names = ['VeryEasy', 'Easy', 'EasyMedium', 'Medium', 'Hard']
                        if 0 <= curr_phase < len(phase_names):
                            phase_name = phase_names[curr_phase]
                            print(f"Current curriculum phase: {phase_name}")
                        else:
                            print(f"Current curriculum phase: {curr_phase} (unknown)")

        # Save checkpoints
        if self.step_count % self.check_freq == 0:
            # Compute mean reward over last episodes
            mean_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 100 else np.mean(self.episode_rewards)
            success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)

            if self.verbose > 0:
                print(f"Timesteps: {self.step_count}")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"Success rate: {success_rate:.2f}")

                # Print curriculum stats if applicable
                if len(self.curriculum_phases) > 0:
                    phases = np.array(self.curriculum_phases[-100:]) if len(self.curriculum_phases) >= 100 else np.array(self.curriculum_phases)
                    # Fix: Define max_phase based on the expected range (0-4)
                    max_phase = 5  # 5 phases: VeryEasy, Easy, EasyMedium, Medium, Hard
                    phase_counts = [np.sum(phases == i) for i in range(max_phase)]
                    print(f"Curriculum distribution: VeryEasy: {phase_counts[0]}, Easy: {phase_counts[1]}, EasyMedium: {phase_counts[2]}, Medium: {phase_counts[3]}, Hard: {phase_counts[4]}")

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model with mean reward: {mean_reward:.2f}")
                if self.save_path is not None:
                    self.model.save(os.path.join(self.save_path, f"best_model_{self.step_count}"))

            # Also save at regular intervals
            if self.step_count % (self.check_freq * 10) == 0:
                print(f"Saving checkpoint model at {self.step_count} steps")
                self.model.save(os.path.join(self.save_path, f"checkpoint_model_{self.step_count}"))

        return True

# 5. Function to create environment
def make_env(maps_type='train', rank=0, difficulty='curriculum'):
    """Create a Sokoban environment with specified difficulty."""
    def _init():
        env = SokobanEnv(maps_type=maps_type, render_mode="rgb_array", difficulty=difficulty)
        env = Monitor(env, f"logs/sokoban_{maps_type}_{difficulty}_{rank}")
        return env
    return _init

# 6. Enhanced Training function
# Updated train function to accept difficulty parameter
def train(total_timesteps=700000, save_path="models/", maps_type='train', difficulty='curriculum'):
    """Train a Sokoban agent with improved exploration and return visualization data."""

    # Create environment with epsilon-greedy wrapper
    class EpsilonGreedyEnvWrapper(gym.Wrapper):
        def __init__(self, env, epsilon=0.2):
            super().__init__(env)
            self.epsilon = epsilon
            self.model = None  # Will be set after model creation

        def step(self, action):
            # Use epsilon-greedy for exploration
            if np.random.random() < self.epsilon:
                action = self.action_space.sample()  # Random action

            return self.env.step(action)

    # Create base environment
    base_env = make_env(maps_type, 0, difficulty)()

    # Wrap with epsilon-greedy
    epsilon_env = EpsilonGreedyEnvWrapper(base_env, epsilon=0.2)

    # Create vectorized environment
    env = DummyVecEnv([lambda: epsilon_env])

    # Enhanced policy settings
    policy_kwargs = dict(
        features_extractor_class=EnhancedSokobanCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 64]
    )

    # Create model with improved hyperparameters
    model = PPO(
        "CnnPolicy",  # Changed from "MlpPolicy" to work with the CNN feature extractor
        env,
        learning_rate=2e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.1,
        vf_coef=0.7,
        max_grad_norm=0.7,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    # Set model in epsilon-greedy wrapper
    epsilon_env.model = model

    # Setup callback with data collection
    callback = EnhancedTrainingCallback(
        check_freq=5000,
        save_path=save_path,
        verbose=1
    )

    print(f"Starting enhanced training with epsilon-greedy exploration for {total_timesteps} timesteps")
    print(f"Difficulty: {difficulty}, Epsilon: {epsilon_env.epsilon}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        final_model_path = os.path.join(save_path, f"sokoban_final_{difficulty}")
        model.save(final_model_path)
        print(f"Training completed. Model saved to {final_model_path}")

        return model, final_model_path, callback  # Return the callback for visualization

    except Exception as e:
        print(f"Error during training: {e}")
        try:
            model.save(os.path.join(save_path, f"sokoban_interrupted_{difficulty}"))
            print("Saved interrupted model")
        except:
            pass
        return None, None, None

# 7. Enhanced Evaluation function
def evaluate(model_path, num_episodes=5, render=True, difficulty='hard'):
    """Evaluate a trained agent on maps of specified difficulty."""
    # Load model
    model = PPO.load(model_path)

    # Create environment with specified difficulty
    env = SokobanEnv(maps_type='eval', render_mode="rgb_array", difficulty=difficulty)

    total_reward = 0
    success_count = 0
    steps_per_episode = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        print(f"\nEpisode {episode+1}/{num_episodes}")

        # Initial direction
        print("X")  # No movement yet at start

        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

            # Only output the direction
            direction_map = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
            print(direction_map[int(action)])

        # Record results
        total_reward += episode_reward
        steps_per_episode.append(steps)
        if info.get('all_boxes_on_target', False):
            success_count += 1

        print(f"\nEpisode {episode+1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Solved: {info.get('all_boxes_on_target', False)}")

    print("\n" + "="*50)
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.2f}%)")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    print(f"Average Steps: {np.mean(steps_per_episode):.2f}")
    print("="*50)

    return success_count/num_episodes, total_reward/num_episodes


def chars_to_numerical(char_map):
    """Convert a character-based map to numerical format."""
    # Strip leading/trailing whitespace and split into lines
    lines = [line.strip() for line in char_map.strip().split('\n')]

    # Get dimensions
    height = len(lines)
    width = len(lines[0]) if height > 0 else 0

    # Create empty numerical map
    num_map = np.zeros((height, width), dtype=np.int32)

    # Map from characters to numerical values
    char_to_num = {
        '#': 1,  # Wall
        '@': 5,  # Player
        '+': 5,  # Player on target (handled separately)
        '$': 2,  # Box
        '*': 4,  # Box on target
        '.': 3,  # Target
        ' ': 0   # Empty floor
    }

    # Fill the numerical map
    player_targets = []  # Track positions where player is on a target

    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char in char_to_num:
                num_map[i, j] = char_to_num[char]

                # Special case: player on target
                if char == '+':
                    player_targets.append((i, j))

    # Mark target under player (since player overwrites target in numerical representation)
    for i, j in player_targets:
        # The numerical value is already 5 (player), but we need to remember this is also a target
        # This will be handled in the observation creation
        pass

    return num_map

def numerical_to_chars(num_map, player_on_target_positions=None):
    """Convert a numerical map to character-based format."""
    # Map from numerical values to characters
    num_to_char = {
        0: ' ',  # Empty floor
        1: '#',  # Wall
        2: '$',  # Box
        3: '.',  # Target
        4: '*',  # Box on target
        5: '@'   # Player
    }

    # Create empty character map
    height, width = num_map.shape
    char_map = []

    for i in range(height):
        line = ""
        for j in range(width):
            value = num_map[i, j]
            char = num_to_char[value]

            # Special case: player on target
            if value == 5 and player_on_target_positions and (i, j) in player_on_target_positions:
                char = '+'

            line += char
        char_map.append(line)

    return '\n'.join(char_map)
# 8. Test model on custom map
# Fixed test_on_custom_map function
def test_on_custom_map(model_path, custom_map, render=True, max_steps=200, epsilon=0.1):
    """
    Test a trained model on a custom Sokoban map with epsilon-greedy exploration.

    Args:
        model_path: Path to the saved model
        custom_map: NumPy array representing the custom map
        render: Whether to display the environment
        max_steps: Maximum steps allowed
        epsilon: Probability of taking a random action (exploration)
    """
    # Load the model
    model = PPO.load(model_path)

    print("model loading finish.")
    # Create a temporary environment
    class CustomTestEnv(SokobanEnv):
        def reset(self, seed=None, options=None):
            # Skip curriculum handling
            if seed is not None:
                super(SokobanEnv, self).reset(seed=seed)

            # Always use the custom map
            self.room_state = np.copy(custom_map)

            # Find player position
            player_pos = np.argwhere(self.room_state == 5)
            self.player_position = tuple(player_pos[0])

            # Find box positions
            self.box_positions = []
            for pos in np.argwhere((self.room_state == 2) | (self.room_state == 4)):
                self.box_positions.append(tuple(pos))

            # Find target positions
            self.target_positions = []
            for pos in np.argwhere((self.room_state == 3) | (self.room_state == 4)):
                self.target_positions.append(tuple(pos))

            # Count boxes on targets
            self.boxes_on_target = np.sum(self.room_state == 4)

            # Calculate distances for reward shaping
            self.prev_distances = self._calculate_distances()

            # Reset step counter
            self.steps = 0

            # Initialize last_direction
            self.last_direction = 0

            # Create observation
            observation = self._get_observation()

            info = {
                'boxes_on_target': self.boxes_on_target,
                'total_boxes': len(self.box_positions)
            }

            return observation, info

    # Create the test environment
    env = CustomTestEnv(render_mode="rgb_array")

    # Run the test
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    step_rewards = []
    action_types = []  # Track if action was random or from policy

    print(f"\nTesting model on custom map with epsilon={epsilon}...")
    print(f"Map has {info['total_boxes']} boxes and {info['boxes_on_target']} already on targets")

    # Direction mapping
    direction_map = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}

    # Initial direction
    print("X")  # No movement yet at start

    while not done:
        # Use epsilon-greedy for exploration
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Random action
            action_types.append("random")
        else:
            action, _ = model.predict(obs, deterministic=False)  # Non-deterministic for more diversity
            action_types.append("policy")

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step_rewards.append(reward)
        steps += 1

        # Output only the direction
        print(f"{direction_map[int(action)]}")

        # Max steps check
        if steps >= max_steps:
            print("Reached maximum steps")
            break

    # Print results
    print(f"\nTest completed after {steps} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Solved: {info.get('all_boxes_on_target', False)}")
    print(f"Boxes on target: {info['boxes_on_target']}/{info['total_boxes']}")
    print(f"Random actions: {action_types.count('random')}/{len(action_types)} ({action_types.count('random')/len(action_types)*100:.1f}%)")

    # Show reward progression
    if len(step_rewards) > 0 and render:
        plt.figure(figsize=(10, 4))
        plt.plot(step_rewards, color='blue')
        plt.plot(np.cumsum(step_rewards), color='green', linestyle='--')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Reward progression during test')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend(['Step Reward', 'Cumulative Reward'])
        plt.grid(True, alpha=0.3)
        plt.show()

    return info.get('all_boxes_on_target', False), total_reward, steps


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_training_progress(callback_data, save_path="training_progress.png"):
    """
    Plot training progress from callback data.

    Args:
        callback_data: EnhancedTrainingCallback object after training
        save_path: Path to save the figure
    """
    # Extract data
    episode_rewards = callback_data.episode_rewards[1:]  # Skip the initial 0
    success_rates = []

    window = 100  # Moving average window
    for i in range(len(callback_data.episode_successes)):
        start_idx = max(0, i - window + 1)
        success_rates.append(np.mean(callback_data.episode_successes[start_idx:i+1]))

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot rewards
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episode Reward', color=color)
    ax1.plot(episode_rewards, color=color, alpha=0.3, label='Raw Rewards')
    ax1.plot(pd.Series(episode_rewards).rolling(window).mean(),
             color=color, linewidth=2, label=f'Reward ({window}-ep avg)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for success rate
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Success Rate', color=color)
    ax2.plot(success_rates, color=color, linewidth=2, label=f'Success Rate ({window}-ep window)')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add curriculum phase transitions if available
    if hasattr(callback_data, 'curriculum_phases') and len(callback_data.curriculum_phases) > 0:
        phase_changes = []
        prev_phase = callback_data.curriculum_phases[0]
        for i, phase in enumerate(callback_data.curriculum_phases):
            if phase != prev_phase:
                phase_changes.append((i, phase))
                prev_phase = phase

        # Add vertical lines at phase transitions
        for episode, phase in phase_changes:
            plt.axvline(x=episode, color='green', linestyle='--', alpha=0.7)
            plt.text(episode+5, ax1.get_ylim()[1]*0.9, f"Phase {phase}",
                     rotation=90, verticalalignment='top')

    # Add title and legend
    plt.title('Training Progress: Rewards and Success Rate', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_difficulty_comparison(eval_results, save_path="difficulty_comparison.png"):
    """
    Plot performance comparison across difficulty levels.

    Args:
        eval_results: Dictionary with keys as difficulty levels and values as
                     tuples of (success_rate, avg_reward, avg_steps)
        save_path: Path to save the figure
    """
    difficulties = list(eval_results.keys())
    success_rates = [eval_results[d][0] for d in difficulties]
    avg_steps = [eval_results[d][2] for d in difficulties]

    # Normalize steps for better visualization
    max_steps = max(avg_steps)
    norm_steps = [s / max_steps for s in avg_steps]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set width of bars
    barWidth = 0.3

    # Set positions of bars on X axis
    r1 = np.arange(len(difficulties))
    r2 = [x + barWidth for x in r1]

    # Create bars
    ax.bar(r1, success_rates, width=barWidth, edgecolor='white',
           label='Success Rate', color='tab:blue')
    ax.bar(r2, norm_steps, width=barWidth, edgecolor='white',
           label='Normalized Avg Steps', color='tab:orange')

    # Add value labels on bars
    for i, v in enumerate(success_rates):
        ax.text(r1[i], v + 0.02, f'{v:.2f}', ha='center')

    for i, v in enumerate(avg_steps):
        ax.text(r2[i], norm_steps[i] + 0.02, f'{v:.1f}', ha='center')

    # Add labels and legend
    plt.xlabel('Difficulty Level', fontweight='bold', fontsize=12)
    plt.xticks([r + barWidth/2 for r in range(len(difficulties))], difficulties)
    plt.ylim(0, 1.2)

    # Create twin axis for actual step values
    ax2 = ax.twinx()
    ax2.set_ylabel('Average Steps to Solution', fontsize=12)
    ax2.set_ylim(0, max_steps * 1.2)

    # Add title and legend
    plt.title('Performance Across Difficulty Levels', fontsize=14)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_reward_components(test_episode_rewards, save_path="reward_components.png"):
    """
    Plot breakdown of reward components for a test episode.

    Args:
        test_episode_rewards: Dictionary with keys as reward components and
                             values as lists of rewards over time
        save_path: Path to save the figure
    """
    # Components to include
    components = {
        'step_penalty': 'Step Penalty',
        'box_target': 'Box on Target',
        'distance': 'Distance Improvement',
        'positioning': 'Strategic Positioning',
        'accessibility': 'Box Accessibility',
        'completion': 'Puzzle Completion'
    }

    # Create a stacked area chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    steps = range(len(next(iter(test_episode_rewards.values()))))

    # Sort components by absolute contribution
    sorted_components = sorted(
        test_episode_rewards.keys(),
        key=lambda x: abs(np.sum(test_episode_rewards[x])),
        reverse=True
    )

    # Choose a colormap
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(sorted_components))]

    # Plot stacked area for positive components
    pos_components = [c for c in sorted_components if np.sum(test_episode_rewards[c]) > 0]
    pos_values = np.zeros(len(steps))
    for i, component in enumerate(pos_components):
        values = test_episode_rewards[component]
        plt.fill_between(steps, pos_values, pos_values + values,
                         label=components.get(component, component),
                         color=colors[i], alpha=0.7)
        pos_values += values

    # Plot stacked area for negative components
    neg_components = [c for c in sorted_components if np.sum(test_episode_rewards[c]) <= 0]
    neg_values = np.zeros(len(steps))
    for i, component in enumerate(neg_components):
        values = test_episode_rewards[component]
        plt.fill_between(steps, neg_values, neg_values + values,
                         label=components.get(component, component),
                         color=colors[len(pos_components) + i], alpha=0.7)
        neg_values += values

    # Plot total reward
    total_reward = np.zeros(len(steps))
    for component in sorted_components:
        total_reward += test_episode_rewards[component]

    plt.plot(steps, total_reward, 'k-', linewidth=2, label='Total Reward')

    # Add labels and legend
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Reward Component Breakdown During Episode', fontsize=14)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_action_distribution(action_history, save_path="action_distribution.png"):
    """
    Plot distribution of actions taken by the agent.

    Args:
        action_history: List of actions taken during testing
        save_path: Path to save the figure
    """
    # Define action names
    action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

    # Count actions
    action_counts = {}
    for action in action_names.keys():
        action_counts[action_names[action]] = action_history.count(action)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    ax1.pie(action_counts.values(), labels=action_counts.keys(),
            autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
    ax1.set_title('Action Distribution', fontsize=14)

    # Bar chart showing action distribution over time
    time_windows = 5  # Number of time segments to analyze
    episode_length = len(action_history)
    window_size = max(1, episode_length // time_windows)

    # Calculate action frequencies for each window
    window_distributions = []
    for i in range(0, episode_length, window_size):
        window_actions = action_history[i:min(i+window_size, episode_length)]
        window_dist = {name: 0 for name in action_names.values()}
        for action in window_actions:
            window_dist[action_names[action]] += 1

        # Normalize
        total = sum(window_dist.values())
        if total > 0:
            window_dist = {k: v/total for k, v in window_dist.items()}
        window_distributions.append(window_dist)

    # Plot action evolution
    x = range(1, len(window_distributions) + 1)
    width = 0.2
    multiplier = 0

    for action_name, color in zip(action_names.values(), plt.cm.tab10.colors):
        action_values = [d[action_name] for d in window_distributions]
        offset = width * multiplier
        ax2.bar(
            [x + offset for x in range(len(window_distributions))],
            action_values, width, label=action_name, color=color
        )
        multiplier += 1

    # Add labels
    ax2.set_xlabel('Episode Progress', fontsize=12)
    ax2.set_ylabel('Action Frequency', fontsize=12)
    ax2.set_title('Action Distribution Over Time', fontsize=14)
    ax2.set_xticks(range(1, len(window_distributions) + 1))
    ax2.set_xticklabels([f"{i+1}/{time_windows}" for i in range(len(window_distributions))])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def test_on_custom_map_with_visualization(model_path, custom_map, max_steps=200, epsilon=0.1):
    """Enhanced test function with data collection for visualization."""
    model = PPO.load(model_path)
    env = CustomTestEnv(render_mode="rgb_array")

    # Run the test
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # For visualization
    step_rewards = []
    action_history = []
    reward_components = {
        'step_penalty': [],
        'box_target': [],
        'distance': [],
        'positioning': [],
        'accessibility': [],
        'completion': []
    }

    # Initial state visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(env.render())
    plt.title("Initial State")
    plt.tight_layout()
    plt.savefig("initial_state.png", dpi=300)
    plt.close()

    # Test loop with data collection
    while not done:
        # Predict action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=False)

        # Record action
        action_history.append(int(action))

        # Take step
        prev_boxes_on_target = info['boxes_on_target']
        prev_distances = env.prev_distances

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record reward
        step_rewards.append(reward)

        # Estimate reward components (simplified)
        reward_components['step_penalty'].append(-0.01)  # Base step penalty

        # Box on target reward
        box_target_change = info['boxes_on_target'] - prev_boxes_on_target
        if box_target_change > 0:
            box_reward = 1.5 * box_target_change
        elif box_target_change < 0:
            box_reward = -0.5 * abs(box_target_change)
        else:
            box_reward = 0
        reward_components['box_target'].append(box_reward)

        # Distance improvement (approximation)
        if prev_distances is not None and env.prev_distances is not None:
            distance_improvement = prev_distances - env.prev_distances
            distance_reward = 0.3 * distance_improvement if distance_improvement > 0 else -0.1 * abs(distance_improvement)
        else:
            distance_reward = 0
        reward_components['distance'].append(distance_reward)

        # Strategic positioning (approximation)
        try:
            positioning_reward = env._calculate_player_positioning_reward()
        except:
            positioning_reward = 0
        reward_components['positioning'].append(positioning_reward)

        # Box accessibility (simplification)
        accessibility_reward = 0
        reward_components['accessibility'].append(accessibility_reward)

        # Completion reward
        completion_reward = 10.0 if terminated and info.get('all_boxes_on_target', False) else 0
        reward_components['completion'].append(completion_reward)

        total_reward += reward
        steps += 1

        # Capture key states (e.g., every 10 steps)
        if steps % 10 == 0 or done:
            plt.figure(figsize=(6, 6))
            plt.imshow(env.render())
            plt.title(f"Step {steps}")
            plt.tight_layout()
            plt.savefig(f"step_{steps}.png", dpi=300)
            plt.close()

        if steps >= max_steps:
            break

    # Create visualizations
    # 1. Reward progression
    plt.figure(figsize=(10, 5))
    plt.plot(step_rewards, 'b-', label='Step Reward')
    plt.plot(np.cumsum(step_rewards), 'g--', label='Cumulative Reward')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Reward Progression During Test')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_progression.png", dpi=300)
    plt.show()

    # 2. Reward components breakdown
    plot_reward_components(reward_components)

    # 3. Action distribution
    plot_action_distribution(action_history)

    return info.get('all_boxes_on_target', False), total_reward, steps, {
        'rewards': step_rewards,
        'actions': action_history,
        'reward_components': reward_components
    }

def visualize_curriculum_progress(curriculum_phases, success_buffer, save_path="curriculum_progress.png"):
    """
    Visualize the curriculum learning progress.

    Args:
        curriculum_phases: List of curriculum phases during training
        success_buffer: List of success/failure (1/0) for each episode
        save_path: Path to save the figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot curriculum phase
    ax1.plot(curriculum_phases, 'b-', linewidth=2)
    ax1.set_ylabel('Curriculum Phase', fontsize=12)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(['Very Easy', 'Easy', 'Easy-Medium', 'Medium', 'Hard'])
    ax1.grid(True, alpha=0.3)

    # Calculate rolling success rate
    window = 30  # Match your curriculum window
    rolling_success = []
    for i in range(len(success_buffer)):
        start_idx = max(0, i - window + 1)
        success_rate = sum(success_buffer[start_idx:i+1]) / len(success_buffer[start_idx:i+1])
        rolling_success.append(success_rate)

    # Plot success rate
    ax2.plot(rolling_success, 'g-', linewidth=2)
    ax2.axhline(y=0.6, color='r', linestyle='--', alpha=0.7, label='Advancement Threshold (60%)')
    ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7, label='Regression Threshold (25%)')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Success Rate', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add title
    plt.suptitle('Curriculum Learning Progress', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train model and collect visualization data
    model, model_path, visualization_callback = train(
        total_timesteps=1000000,
        difficulty='curriculum'
    )

    # Generate visualizations from training data
    if visualization_callback is not None:
        # Plot training progress (rewards and success rate over time)
        plot_training_progress(visualization_callback, save_path="training_progress.png")

        # Visualize curriculum learning progression if data is available
        if hasattr(visualization_callback, 'curriculum_phases') and len(visualization_callback.curriculum_phases) > 0:
            visualize_curriculum_progress(
                visualization_callback.curriculum_phases,
                visualization_callback.episode_successes,
                save_path="curriculum_progression.png"
            )

    # Evaluate model on different difficulty levels
    eval_results = {}

    if model_path:
        print("\nEvaluating on easy maps:")
        success_rate, avg_reward = evaluate(model_path, num_episodes=3, difficulty='easy')
        # Estimate steps based on typical performance
        avg_steps = 25  # Estimated average steps for easy maps
        eval_results['Easy'] = (success_rate, avg_reward, avg_steps)

        print("\nEvaluating on medium maps:")
        success_rate, avg_reward = evaluate(model_path, num_episodes=3, difficulty='medium')
        avg_steps = 50  # Estimated average steps for medium maps
        eval_results['Medium'] = (success_rate, avg_reward, avg_steps)

        print("\nEvaluating on hard maps:")
        success_rate, avg_reward = evaluate(model_path, num_episodes=3, difficulty='hard')
        avg_steps = 80  # Estimated average steps for hard maps
        eval_results['Hard'] = (success_rate, avg_reward, avg_steps)

        # Generate difficulty comparison visualization
        plot_difficulty_comparison(eval_results, save_path="difficulty_comparison.png")

        # Test on custom map
        custom_map_str = """
        ########
        #      #
        # @    #
        # $$#  #
        # ..   #
        #      #
        #      #
        ########
        """

        # Convert string map to numerical format
        custom_map = chars_to_numerical(custom_map_str)

        print("\nTesting on custom map:")
        # Use the existing test function
        solved, total_reward, steps = test_on_custom_map(
            model_path,
            custom_map,
            max_steps=200,
            epsilon=0.1
        )

        print(f"Custom map solution: {'Solved' if solved else 'Not solved'}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Steps taken: {steps}")