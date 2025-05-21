# Star Pusher Game Spec

## How to Run

1. Run the command: `pip install -r requirements.txt`
2. run the command: `python3 starpusher.py` <br>
   First time running will download the dataset from a git repository, which might take less than 1 minutes.

Human playing functionality works well, but AI assistant functionality needs to wait for the model ready.

## How to enter training mode

1. open starpusher.py
2. change `mode = "training` (around line 11)
3. run the command `python3 starpusher.py`
4. your actions will be recorded in the folder `sokoban_cache/human_training`

- if you are not satisfied with your solution, press key r to reset the current game
- if you replay the game, the new actions will replace the original recording.
- if you do not solve the level successfully, your actions for this level will not be recorded.
- if you press an arrow key but the player could not move in that direction, your invalid action will not be recorded.
- if you terminate the game, the recording for the previous levels will not be affected. The current un-finished actions will not be recorded.
- If you choose the default Learning level, all the curriculum learning maps will be loaded.
- If you choose other levels, e.g. Medium or Hard, each time 20 randomly chosen maps will be loaded.
- If you are not satisfied with your recording, but the recording has been saved to the file, you could find the recording .txt file name in the Terminal window and then manually delete it from `human_training` folder.
- different persons's training result could be put in the same folder if the file names are unique.
- In training mode, you could not use AI Assistant mode.

## How to enter Play mode

1. open starpusher.py
2. change `mode="play"` (around line 11)
3. run the command `python3 starpusher.py`
4. your actions will NOT be recorded.
5. Click AI Assistent button to switch to AI mode. Wait for the AI to give you the solution and it will move along its route.
6. Anytime, you could switch back to human play mode by clicking "Human Play" button.
