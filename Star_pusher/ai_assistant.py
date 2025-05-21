import numpy as np 
from stable_baselines3 import PPO
import model_related
import os

######## convertion rules ############
    # 0: " ",
    # 1: "#",
    # 2: "$",
    # 3: ".",
    # 4: "*",
    # 5: "@"
######################################
    
parent_dir = os.path.dirname(os.path.realpath(__file__))
model_path = parent_dir + "/models/sokoban_final_curriculum.zip"

if not os.path.exists(model_path):
    print(f"The file {model_path} does not exist.")
   

def convert_to_ai_map(starting_map_info, stars_pos, player_pos):
    map_width = starting_map_info['width']
    map_height = starting_map_info['height']
    ai_map = np.zeros((map_height, map_width))
    
    # print("starting mpa info:", starting_map_info)
    # print("stars_pos:", stars_pos)
    # print("player_pos:", player_pos)
    
    starting_map = starting_map_info['mapObj']
    goal_pos = starting_map_info['goals']
    
    for i in range(map_height):
        for j in range(map_width):
            if starting_map[i][j] == "#":
                ai_map[i, j] = 1
            # updated player and stars position
            elif (i, j) == player_pos:
                ai_map[i][j] = 5
            elif (i, j) in stars_pos:
                if (i, j) in goal_pos:
                    ai_map[i][j] = 4
                else:
                    ai_map[i][j] = 2
            elif starting_map[i][j] == ".":
                ai_map[i, j] = 3
            else:
            # other places are just floor
                ai_map[i][j] = 0

    # print(ai_map)
    return ai_map


def get_ai_actions(ai_map):

    # result = model_related.test_on_custom_map(model_path, ai_map, False)
    result = "LRRD"
    actions = []
    for char in result: 
        if char == "L":
            actions.append("left")
        elif char == "R":
            actions.append("right")
        elif char == "U":
            actions.append("up")
        elif char == "D":
            actions.append("down")
    
    return actions