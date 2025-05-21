# Updated version based on the code from Al Sweigart al@inventwithpython.com

import random, sys, copy, os, pygame
from pygame.locals import *
import prepare_dataset
import ai_assistant
import multiprocessing
import math


mode = "training"
FPS = 30 # frames per second to update the screen
WINWIDTH = 1000 # width of the program's window, in pixels
WINHEIGHT = 600 # height in pixels
HALF_WINWIDTH = int(WINWIDTH / 2)
HALF_WINHEIGHT = int(WINHEIGHT / 2)

# The total width and height of each tile in pixels.
TILEWIDTH = 50
TILEHEIGHT = 50

# The percentage of outdoor tiles that have additional
# decoration on them, such as a tree or rock.
OUTSIDE_DECORATION_PCT = 35

BRIGHTBLUE = (  0, 170, 255)
WHITE      = (255, 255, 255)
BGCOLOR = BRIGHTBLUE
TEXTCOLOR = WHITE

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

current_difficulty = "learning"
parent_dir = os.path.dirname(os.path.realpath(__file__))
cache_path = parent_dir + "/sokoban_cache"
human_training_path = cache_path + "/human_training"
human_training_actions = ""
ai_action_process = None

def process_ai_inference(data, queue):
    result = ai_assistant.get_ai_actions(data)
    queue.put(result)
 
# def draw_spinner(surface, center, radius, angle, color=(255, 255, 255), width=4):
#     x = center[0] + radius * math.cos(angle)
#     y = center[1] + radius * math.sin(angle)
#     pygame.draw.line(surface, color, center, (x, y), width)    

# when mode == "play", will call this function to read 5 maps
def read_maps_from_file(data_folder):
    
    if mode == "training" and current_difficulty == "learning":
        selected_maps, maps_file_name = prepare_dataset.choose_all_maps(data_folder)
    else:
        maps, source_file = prepare_dataset.select_maps(data_folder)
        partial_path = source_file[len(cache_path)+1:].replace("/", "_")[:-4] 
        print("source_file:", source_file)
        map_num = min(len(maps), 20)
        selected_maps = random.sample(maps, map_num)
        
        maps_file_name = []
        for map in selected_maps:
            maps_file_name.append(partial_path + "_" + str(maps.index(map)) + ".txt")
                
    levels = []
    for mapTextLines in selected_maps:
        level = read_map(mapTextLines)
        levels.append(level)
         
        ### delete later. now for debugging
        # print("just test if the conversion matches: this one from Sarjunes!")
        # custom_map = model_related.chars_to_numerical("\n".join(mapTextLines))       
        # print(custom_map)
                     
    return levels, maps_file_name
    
    
def main():
    global FPSCLOCK, DISPLAYSURF, IMAGESDICT, TILEMAPPING, OUTSIDEDECOMAPPING, BASICFONT, PLAYERIMAGES, currentImage, ROBOTIMAGE, ai_mode, human_training_actions

    # Pygame initialization and basic set up of the global variables.
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    

    # Because the Surface object stored in DISPLAYSURF was returned
    # from the pygame.display.set_mode() function, this is the
    # Surface object that is drawn to the actual computer screen
    # when pygame.display.update() is called.
    DISPLAYSURF = pygame.display.set_mode((WINWIDTH, WINHEIGHT))

    pygame.display.set_caption('Star Pusher')
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)

    # A global dict value that will contain all the Pygame
    # Surface objects returned by pygame.image.load().
    IMAGESDICT = {'uncovered goal': pygame.transform.scale(pygame.image.load('images/'+ 'RedSelector.png'), (50,50)),
                  'covered goal': pygame.transform.scale(pygame.image.load('images/'+ 'Selector.png'), (50,50)),
                  'star': pygame.transform.scale(pygame.image.load('images/'+ 'Star.png'), (40,40)),
                  'corner': pygame.transform.scale(pygame.image.load('images/'+ 'Wall_Block_Tall.png'), (55,55)),
                  'wall': pygame.transform.scale(pygame.image.load('images/'+ 'Wood_Block_Tall.png'), (50, 50)),
                  'inside floor': pygame.image.load('images/'+ 'Plain_Block.png'),
                  'outside floor': pygame.transform.scale(pygame.image.load('images/'+ 'Grass_Block.png'), (50,60)),
                  'title': pygame.image.load('images/'+ 'star_title.png'),
                  'solved': pygame.image.load('images/'+ 'star_solved.png'),
                  'princess': pygame.transform.scale(pygame.image.load('images/'+ 'princess.png'), (40, 45)),
                  'boy': pygame.transform.scale(pygame.image.load('images/'+ 'boy.png'), (40,45)),
                  'catgirl': pygame.transform.scale(pygame.image.load('images/'+ 'catgirl.png'), (40,45)),
                  'pinkgirl': pygame.transform.scale(pygame.image.load('images/'+ 'pinkgirl.png'), (40,45)),
                  'rock': pygame.transform.scale(pygame.image.load('images/'+ 'Rock.png'), (35, 35)),
                  'short tree': pygame.transform.scale(pygame.image.load('images/'+ 'Tree_Short.png'), (35,35)),
                  'tall tree': pygame.transform.scale(pygame.image.load('images/'+ 'Tree_Tall.png'),(50, 35)),
                  'ugly tree': pygame.transform.scale(pygame.image.load('images/'+ 'Tree_Ugly.png'), (45, 40)),
                  'ai_assistant': pygame.image.load('images/'+ 'ai_assistant.png'),
                  'ai_assistant_no_bg': pygame.image.load('images/'+ 'ai_assistant_no_bg.png'),
                  'human_play': pygame.image.load('images/'+ 'human_play.png'),
                  'human_play_no_bg': pygame.image.load('images/'+ 'human_play_no_bg.png'),
                  'learning': pygame.image.load('images/'+ 'learning.png'),
                  'learning_highlight': pygame.image.load('images/'+ 'learning_highlight.png'),
                  'medium': pygame.image.load('images/'+ 'medium.png'),
                  'medium_highlight': pygame.image.load('images/'+ 'medium_highlight.png'),
                  'hard': pygame.image.load('images/'+ 'hard.png'),
                  'hard_highlight': pygame.image.load('images/'+ 'hard_highlight.png'),
                  "robot": pygame.transform.scale(pygame.image.load('images/'+ 'robot-9.png'), (50, 50))}

    # These dict values are global, and map the character that appears
    # in the level file to the Surface object it represents.
    TILEMAPPING = {'x': IMAGESDICT['corner'],
                   '#': IMAGESDICT['wall'],
                   'o': IMAGESDICT['inside floor'],
                   ' ': IMAGESDICT['outside floor']}
    OUTSIDEDECOMAPPING = {'1': IMAGESDICT['rock'],
                          '2': IMAGESDICT['short tree'],
                          '3': IMAGESDICT['tall tree'],
                          '4': IMAGESDICT['ugly tree']}

    # PLAYERIMAGES is a list of all possible characters the player can be.
    # currentImage is the index of the player's current player image.
    currentImage = 0
    PLAYERIMAGES = [IMAGESDICT['princess'],
                    IMAGESDICT['boy'],
                    IMAGESDICT['catgirl'],
                    IMAGESDICT['pinkgirl']]
    ROBOTIMAGE = IMAGESDICT['robot']

    startScreen() # show the title screen until the user presses a key

    # when the mode is training, we need to read all maps from the folder to train continuously
    if current_difficulty == "learning":
        levels, maps_file_name = read_maps_from_file(cache_path + '/Curriculum-levels')

    elif current_difficulty == "medium":
        levels, maps_file_name = read_maps_from_file(cache_path + '/boxoban-levels-master/medium/train')

    elif current_difficulty == "hard":
        levels, maps_file_name = read_maps_from_file(cache_path + '/boxoban-levels-master/hard')
            
    if mode == "training":
        os.makedirs(human_training_path, exist_ok=True)
        
    currentLevelIndex = 0
    ai_mode = False
    
    # The main game loop. This loop runs a single level, when the user
    # finishes that level, the next/previous level is loaded.
    while True: # main game loop
        # Run the level to actually start playing the game:
        result = runLevel(levels, currentLevelIndex, ai_mode)

        if result in ('solved', 'next'):
            # Go to the next level.

            if result == "solved" and mode == "training":
                saved_to_file = human_training_path + "/" + maps_file_name[currentLevelIndex]
                prepare_dataset.export_to_txt_file(saved_to_file, levels[currentLevelIndex]['mapObj'], human_training_actions)
                print("training result is saved to:", saved_to_file)
                human_training_actions = ""
                
            currentLevelIndex += 1
            if currentLevelIndex >= len(levels):
                # If there are no more levels, go back to the first one.
                currentLevelIndex = 0
            
                    
        elif result == 'back':
            # Go to the previous level.
            currentLevelIndex -= 1
            if currentLevelIndex < 0:
                # If there are no previous levels, go to the last one.
                currentLevelIndex = len(levels)-1
        elif result == 'reset':
            pass # Do nothing. Loop re-calls runLevel() to reset the level



def runLevel(levels, levelNum, ai_mode=False):
    global currentImage, human_training_actions, ai_action_process
    
    levelObj = levels[levelNum]
    mapObj = decorateMap(levelObj['mapObj'], levelObj['startState']['player'])
    gameStateObj = copy.deepcopy(levelObj['startState'])
    mapNeedsRedraw = True # set to True to call drawMap()
    levelSurf = BASICFONT.render('Level %s of %s' % (levelNum + 1, len(levels)), 1, TEXTCOLOR)
    levelRect = levelSurf.get_rect()
    levelRect.bottomleft = (86, WINHEIGHT - 20)
    levelIsComplete = False
    
    
    if current_difficulty == "learning":
        difficulty_surf = IMAGESDICT['learning_highlight']
        difficulty_surf = pygame.transform.scale(difficulty_surf, (102, 66))
    elif current_difficulty == "medium":
        difficulty_surf = IMAGESDICT['medium_highlight']
        difficulty_surf = pygame.transform.scale(difficulty_surf, (102, 66))
    elif current_difficulty == "hard":
        difficulty_surf = IMAGESDICT['hard_highlight']
        difficulty_surf = pygame.transform.scale(difficulty_surf, (102, 66))

    difficulty_rect = difficulty_surf.get_rect()
    difficulty_rect.top = WINHEIGHT - 56
    difficulty_rect.centerx = WINWIDTH - 60
    
    
    # TBD: just mock data here. In future, ai_actions will be replaced by model's inference
    # ai_actions = [LEFT, RIGHT, RIGHT, LEFT, UP, DOWN, DOWN]
    ai_actions = []
    ai_step = 0
    ai_speed_interval = 500
    human_playRect = None
    ai_assistantRect = None
    ai_action_loading = False 

    while True: # main game loop
        # Reset these variables:
        playerMoveTo = None
        keyPressed = False

        if ai_mode:
            # when ai_action is loading, wait for the result_queue
            if ai_action_loading:
                if not result_queue.empty():
                    ai_actions = result_queue.get()
                    ai_action_loading = False
                    ai_step = 0
                    
                    print("retrive action from ai:", ai_actions)
            else: 
                for event in pygame.event.get(): # event handling loop
                    if event.type == QUIT:
                        # Player clicked the "X" at the corner of the window.
                        
                        terminate(ai_action_process)
                        
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if human_playRect is not None and human_playRect.collidepoint(event.pos):
                            ai_mode = False  # Change variable if image1 is clicked
                            mapNeedsRedraw = True
                
                if ai_actions is not None and len(ai_actions) > 0: 
                    if ai_speed_interval < 0 and ai_step < len(ai_actions):
                        playerMoveTo = ai_actions[ai_step]
                        ai_speed_interval = 1000
                        ai_step += 1
                        
                    else:
                        playerMoveTo = None
                    ai_speed_interval -= 1
                
        elif not ai_mode:
            for event in pygame.event.get(): # event handling loop
                if event.type == QUIT:
                    # Player clicked the "X" at the corner of the window.
                    terminate(ai_action_process)
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if mode != "training":
                        if ai_assistantRect is not None and ai_assistantRect.collidepoint(event.pos):
                            ai_map = ai_assistant.convert_to_ai_map(levelObj, gameStateObj['stars'], gameStateObj['player'])
                            
                            result_queue = multiprocessing.Queue()
                            ai_action_process = multiprocessing.Process(target=process_ai_inference, args=(ai_map, result_queue))
                            ai_action_process.start()
                            ai_action_loading = True
                            ai_actions = None
                            ai_mode = True  # Change variable if image2 is clicked
                            mapNeedsRedraw = True
                        
                        
                if event.type == KEYDOWN:
                    # Handle key presses
                    keyPressed = True
                    if event.key == K_LEFT:
                        playerMoveTo = LEFT
                    elif event.key == K_RIGHT:
                        playerMoveTo = RIGHT
                    elif event.key == K_UP:
                        playerMoveTo = UP
                    elif event.key == K_DOWN:
                        playerMoveTo = DOWN

                    elif event.key == K_n:
                        human_training_actions = ""
                        return 'next'
                    elif event.key == K_b:
                        human_training_actions = ""
                        return 'back'

                    elif event.key == K_ESCAPE:
                        terminate(ai_action_process) # Esc key quits.
                    elif event.key == K_r:
                        human_training_actions = ""
                        return 'reset' # Reset the level.
                    elif event.key == K_p:
                        # Change the player image to the next one.
                        currentImage += 1
                        if currentImage >= len(PLAYERIMAGES):
                            # After the last player image, use the first one.
                            currentImage = 0
                        mapNeedsRedraw = True

        if playerMoveTo is not None and not levelIsComplete:
            # If the player pushed a key to move, make the move
            # (if possible) and push any stars that are pushable.
            moved = makeMove(mapObj, gameStateObj, playerMoveTo)

            if moved:
                # increment the step counter.
                gameStateObj['stepCounter'] += 1
                mapNeedsRedraw = True

            if isLevelFinished(levelObj, gameStateObj):
                # level is solved, we should show the "Solved!" image.
                levelIsComplete = True
                keyPressed = False

        DISPLAYSURF.fill(BGCOLOR)

        if mapNeedsRedraw:
            mapSurf = drawMap(mapObj, gameStateObj, levelObj['goals'], ai_mode)
            mapNeedsRedraw = False

        mapSurfRect = mapSurf.get_rect()
        mapSurfRect.center = (2 * HALF_WINWIDTH // 3, HALF_WINHEIGHT)

        # Draw mapSurf to the DISPLAYSURF Surface object.
        DISPLAYSURF.blit(mapSurf, mapSurfRect)
        DISPLAYSURF.blit(levelSurf, levelRect)
        ai_assistantRect, human_playRect = draw_button_option(ai_mode)
        draw_spec()
        
        stepSurf = BASICFONT.render('Steps: %s' % (gameStateObj['stepCounter']), 1, TEXTCOLOR)
        stepRect = stepSurf.get_rect()
        stepRect.bottomleft = (220, WINHEIGHT - 20)
        DISPLAYSURF.blit(stepSurf, stepRect)
        
        DISPLAYSURF.blit(difficulty_surf, difficulty_rect)
        
        if levelIsComplete:
            # is solved, show the "Solved!" image until the player
            # has pressed a key.
            solvedRect = IMAGESDICT['solved'].get_rect()
            solvedRect.center = (HALF_WINWIDTH, HALF_WINHEIGHT)
            DISPLAYSURF.blit(IMAGESDICT['solved'], solvedRect)

            if keyPressed:
                return 'solved'
            
        
        pygame.display.update() # draw DISPLAYSURF to the screen.
        FPSCLOCK.tick()


def isWall(mapObj, x, y):
    """Returns True if the (x, y) position on
    the map is a wall, otherwise return False."""
    if x < 0 or x >= len(mapObj) or y < 0 or y >= len(mapObj[x]):
        return False # x and y aren't actually on the map.
    elif mapObj[x][y] in ('#', 'x'):
        return True # wall is blocking
    return False


def decorateMap(mapObj, startxy):
    """Makes a copy of the given map object and modifies it.
    Here is what is done to it:
        * Walls that are corners are turned into corner pieces.
        * The outside/inside floor tile distinction is made.
        * Tree/rock decorations are randomly added to the outside tiles.

    Returns the decorated map object."""

    startx, starty = startxy # Syntactic sugar

    # Copy the map object so we don't modify the original passed
    mapObjCopy = copy.deepcopy(mapObj)

    # Remove the non-wall characters from the map data
    for x in range(len(mapObjCopy)):
        for y in range(len(mapObjCopy[0])):
            if mapObjCopy[x][y] in ('$', '.', '@', '+', '*'):
                mapObjCopy[x][y] = ' '

    # Flood fill to determine inside/outside floor tiles.
    floodFill(mapObjCopy, startx, starty, ' ', 'o')
    
    row_num = len(mapObjCopy)
    col_num = len(mapObjCopy[0])
    #  find the decoration area beyond the left boundary
    for x in range(row_num):
        for y in range(col_num):
            if mapObjCopy[x][y] == '#':
                if x == 0 and \
                   isWall(mapObj, x, y+1) and \
                   isWall(mapObj, x+1, y):
                    mapObjCopy[x][y] = " "
                # indicate it is beyond the wall boundary
                elif x == row_num-1 and \
                    isWall(mapObj, x, y+1) and \
                    isWall(mapObj, x-1, y):
                    mapObjCopy[x][y] = " "
                elif isWall(mapObj, x, y+1) and \
                   isWall(mapObj, x-1, y) and \
                   isWall(mapObj, x+1, y):
                    mapObjCopy[x][y] = " "
                else:
                    break
                    
    # find the decoration area beyond the right boundary              
    for x in range(row_num):
        for y in range(col_num-1, -1, -1):
            if mapObjCopy[x][y] == '#':
                if x == 0 and \
                   isWall(mapObj, x, y-1) and \
                   isWall(mapObj, x+1, y):
                    mapObjCopy[x][y] = " "
                # indicate it is beyond the wall boundary
                elif x == row_num-1 and \
                    isWall(mapObj, x, y-1) and \
                    isWall(mapObj, x-1, y):
                    mapObjCopy[x][y] = " "
                elif isWall(mapObj, x, y-1) and \
                   isWall(mapObj, x-1, y) and \
                   isWall(mapObj, x+1, y):
                    mapObjCopy[x][y] = " "
                else:
                    break
    
    for x in range(row_num):
        for y in range(col_num):     
            if mapObjCopy[x][y] == ' ': 
                if  random.randint(0, 99) < OUTSIDE_DECORATION_PCT:
                    mapObjCopy[x][y] = random.choice(list(OUTSIDEDECOMAPPING.keys()))
                    
                if (isWall(mapObjCopy, x, y-1) and isWall(mapObjCopy, x+1, y)) or \
                    (isWall(mapObjCopy, x+1, y) and isWall(mapObjCopy, x, y+1)) or \
                    (isWall(mapObjCopy, x, y+1) and isWall(mapObjCopy, x-1, y)) or \
                    (isWall(mapObjCopy, x-1, y) and isWall(mapObjCopy, x, y-1)):
                    mapObjCopy[x][y] = 'x'
                        
    return mapObjCopy


def isBlocked(mapObj, gameStateObj, x, y):
    """Returns True if the (x, y) position on the map is
    blocked by a wall or star, otherwise return False."""

    if isWall(mapObj, x, y):
        return True

    elif x < 0 or x >= len(mapObj) or y < 0 or y >= len(mapObj[x]):
        return True # x and y aren't actually on the map.

    elif (x, y) in gameStateObj['stars']:
        return True # a star is blocking

    return False


def makeMove(mapObj, gameStateObj, playerMoveTo):
    """Given a map and game state object, see if it is possible for the
    player to make the given move. If it is, then change the player's
    position (and the position of any pushed star). If not, do nothing.

    Returns True if the player moved, otherwise False."""
    global human_training_actions

    # Make sure the player can move in the direction they want.
    playerx, playery = gameStateObj['player']

    # This variable is "syntactic sugar". Typing "stars" is more
    # readable than typing "gameStateObj['stars']" in our code.
    stars = gameStateObj['stars']

    # The code for handling each of the directions is so similar aside
    # from adding or subtracting 1 to the x/y coordinates. We can
    # simplify it by using the xOffset and yOffset variables.
    if playerMoveTo == UP:
        xOffset = -1
        yOffset = 0
        if mode == "training":
            human_training_actions += "U"
    elif playerMoveTo == RIGHT:
        xOffset = 0
        yOffset = 1
        if mode == "training":
            human_training_actions += "R"
    elif playerMoveTo == DOWN:
        xOffset = 1
        yOffset = 0
        if mode == "training":
            human_training_actions += "D"
    elif playerMoveTo == LEFT:
        xOffset = 0
        yOffset = -1
        if mode == "training":
            human_training_actions += "L"

    # See if the player can move in that direction.
    if isWall(mapObj, playerx + xOffset, playery + yOffset):
        if mode == "training":
            human_training_actions = human_training_actions[:-1]
        return False
    else:
        if (playerx + xOffset, playery + yOffset) in stars:
            # There is a star in the way, see if the player can push it.
            if not isBlocked(mapObj, gameStateObj, playerx + (xOffset*2), playery + (yOffset*2)):
                # Move the star.
                ind = stars.index((playerx + xOffset, playery + yOffset))
                stars[ind] = (stars[ind][0] + xOffset, stars[ind][1] + yOffset)
            else:
                if mode == "training":
                    human_training_actions = human_training_actions[:-1]
                return False
        # Move the player upwards.
        gameStateObj['player'] = (playerx + xOffset, playery + yOffset)
        return True

def draw_button_option(ai_mode):
    """Display the start screen (which has the title and instructions)
    until the player presses a key. Returns None."""
    global DISPLAYSURF
    
    if ai_mode:
        
        ai_optionRect = IMAGESDICT['ai_assistant'].get_rect()
        ai_optionRect.centerx = WINWIDTH - 200
        ai_optionRect.centery = 80
        if mode == "play":
            DISPLAYSURF.blit(IMAGESDICT['ai_assistant'], ai_optionRect)
        
        human_optionRect = IMAGESDICT['human_play_no_bg'].get_rect()
        human_optionRect.centerx = WINWIDTH - 200
        human_optionRect.centery = 140
        DISPLAYSURF.blit(IMAGESDICT['human_play_no_bg'], human_optionRect)
    else:
        ai_optionRect = IMAGESDICT['ai_assistant_no_bg'].get_rect()
        ai_optionRect.centerx = WINWIDTH - 200
        ai_optionRect.centery = 80
        if mode == "play":
            DISPLAYSURF.blit(IMAGESDICT['ai_assistant_no_bg'], ai_optionRect)
        
        human_optionRect = IMAGESDICT['human_play'].get_rect()
        human_optionRect.centerx = WINWIDTH - 200
        human_optionRect.centery = 140
        DISPLAYSURF.blit(IMAGESDICT['human_play'], human_optionRect)
    return ai_optionRect, human_optionRect

def draw_spec():
    """Display the game specification"""

    instructionText = ['key b: previous level',
                       'Key n: next level',
                       'Key p: change costume',
                       'key r: reset current game',
                       'Esc: terminate']

    topCoord = 3 * WINHEIGHT // 4
    # Position and draw the text.
    for i in range(len(instructionText)):
        instSurf = BASICFONT.render(instructionText[i], 1, TEXTCOLOR)
        instRect = instSurf.get_rect()
        instRect.top = topCoord
        instRect.left = WINWIDTH - 300
        topCoord += instRect.height # Adjust for the height of the line.
        DISPLAYSURF.blit(instSurf, instRect)
        

def startScreen():
    """Display the start screen (which has the title and instructions)
    until the player presses a key. Returns None."""
    global current_difficulty
    
    # Position the title image.
    titleRect = IMAGESDICT['title'].get_rect()
    topCoord = 50 # topCoord tracks where to position the top of the text
    titleRect.top = topCoord
    titleRect.centerx = HALF_WINWIDTH
    topCoord += titleRect.height
    

    # Unfortunately, Pygame's font & text system only shows one line at
    # a time, so we can't use strings with \n newline characters in them.
    # So we will use a list with each line in it.
    instructionText = ['Push the stars over the marks.',
                       'Choose different levels.',
                       "Use AI assistant."]

    # Start with drawing a blank color to the entire window:
    DISPLAYSURF.fill(BGCOLOR)
    
    # Position the title image.
    learning_rect = IMAGESDICT['learning'].get_rect()
    learning_rect.top = topCoord + 80
    learning_rect.centerx = 250
    learning_highlight_rect = IMAGESDICT['learning_highlight'].get_rect()
    learning_highlight_rect.top = topCoord + 80
    learning_highlight_rect.centerx = 250
    
    medium_rect = IMAGESDICT['medium'].get_rect()
    medium_rect.top = topCoord + 80
    medium_rect.centerx = 500
    medium_highlight_rect = IMAGESDICT['medium_highlight'].get_rect()
    medium_highlight_rect.top = topCoord + 80
    medium_highlight_rect.centerx = 500
    
    hard_rect = IMAGESDICT['hard'].get_rect()
    hard_rect.top = topCoord + 80
    hard_rect.centerx = 750
    hard_highlight_rect = IMAGESDICT['hard_highlight'].get_rect()
    hard_highlight_rect.top = topCoord + 80
    hard_highlight_rect.centerx = 750
    

    # Draw the title image to the window:
    DISPLAYSURF.blit(IMAGESDICT['title'], titleRect)
    
    if current_difficulty == "learning":
        DISPLAYSURF.blit(IMAGESDICT['learning_highlight'], learning_highlight_rect)
        DISPLAYSURF.blit(IMAGESDICT['medium'], medium_rect)
        DISPLAYSURF.blit(IMAGESDICT['hard'], hard_rect)
    elif current_difficulty == "medium":
        DISPLAYSURF.blit(IMAGESDICT['learning'], learning_rect)
        DISPLAYSURF.blit(IMAGESDICT['medium_highlight'], medium_highlight_rect)
        DISPLAYSURF.blit(IMAGESDICT['hard'], hard_rect)
    elif current_difficulty == "hard":
        DISPLAYSURF.blit(IMAGESDICT['learning'], learning_rect)
        DISPLAYSURF.blit(IMAGESDICT['medium'], medium_rect)
        DISPLAYSURF.blit(IMAGESDICT['hard_highlight'], hard_highlight_rect)

    # Position and draw the text.
    for i in range(len(instructionText)):
        instSurf = BASICFONT.render(instructionText[i], 1, TEXTCOLOR)
        instRect = instSurf.get_rect()
        topCoord += 10 # 10 pixels will go in between each line of text.
        instRect.top = topCoord
        instRect.centerx = HALF_WINWIDTH
        topCoord += instRect.height # Adjust for the height of the line.
        DISPLAYSURF.blit(instSurf, instRect)

    start_delay = 8
    level_chosen = False
    while True: # Main loop for the start screen.
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate(ai_action_process)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    terminate(ai_action_process)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if learning_rect.collidepoint(event.pos) or learning_highlight_rect.collidepoint(event.pos):
                    level_chosen = True
                    current_difficulty = "learning"
                    DISPLAYSURF.blit(IMAGESDICT['learning_highlight'], learning_highlight_rect)
                    DISPLAYSURF.blit(IMAGESDICT['medium'], medium_rect)
                    DISPLAYSURF.blit(IMAGESDICT['hard'], hard_rect)

                elif medium_rect.collidepoint(event.pos) or medium_highlight_rect.collidepoint(event.pos):
                    start_delay -= 1
                    level_chosen = True
                    current_difficulty = "medium"
                    DISPLAYSURF.blit(IMAGESDICT['learning'], learning_rect)
                    DISPLAYSURF.blit(IMAGESDICT['medium_highlight'], medium_highlight_rect)
                    DISPLAYSURF.blit(IMAGESDICT['hard'], hard_rect)
                    
                elif hard_rect.collidepoint(event.pos) or hard_highlight_rect.collidepoint(event.pos):
                    start_delay -= 1
                    level_chosen = True
                    current_difficulty = "hard"
                    DISPLAYSURF.blit(IMAGESDICT['learning'], learning_rect)
                    DISPLAYSURF.blit(IMAGESDICT['medium'], medium_rect)
                    DISPLAYSURF.blit(IMAGESDICT['hard_highlight'], hard_highlight_rect)
                else:
                    level_chosen = True 
                    current_difficulty = "learning"

                
            if level_chosen:
                start_delay -= 1
                if start_delay < 0:
                    return # user has pressed a key or mouse click, so return.

        # Display the DISPLAYSURF contents to the actual screen.
        pygame.display.update()
        FPSCLOCK.tick()


import model_related
# map_data is stored as list. Each item represents one line's layout
def read_map(mapTextLines):
    
    mapObj = [list(mapline) for mapline in mapTextLines]     

    # Loop through the spaces in the map and find the @, ., and $
    # characters for the starting game state.
    startx = None # The x and y for the player's starting position
    starty = None
    goals = [] # list of (x, y) tuples for each goal.
    stars = [] # list of (x, y) for each star's starting position.

    for x in range(len(mapObj)):
        for y in range(len(mapObj[0])):

            if mapObj[x][y] == '@':
                startx = x
                starty = y

            if mapObj[x][y] == '.':
                goals.append((x, y))
            if mapObj[x][y] == '$':
                stars.append((x, y))
            if mapObj[x][y] == "*":
                goals.append((x, y))
                stars.append((x, y))

    # Basic level design sanity checks:
    assert startx != None and starty != None, 'Level missing a "@" or "+" to mark the start point.' 
    assert len(goals) > 0, 'Levelmust have at least one goal.'
    assert len(stars) >= len(goals), 'Level is impossible to solve. It has %s goals but only %s stars.'

    # Create level object and starting game state object.
    gameStateObj = {'player': (startx, starty),
                    'stepCounter': 0,
                    'stars': stars}
    levelObj = {'width': len(mapObj[0]),
                'height': len(mapObj),
                'mapObj': mapObj,
                'goals': goals,
                'startState': gameStateObj}

    return levelObj


def floodFill(mapObj, x, y, oldCharacter, newCharacter):
    """Changes any values matching oldCharacter on the map object to
    newCharacter at the (x, y) position, and does the same for the
    positions to the left, right, down, and up of (x, y), recursively."""

    # In this game, the flood fill algorithm creates the inside/outside
    # floor distinction. This is a "recursive" function.
    # For more info on the Flood Fill algorithm, see:
    #   http://en.wikipedia.org/wiki/Flood_fill
    if mapObj[x][y] == oldCharacter:
        mapObj[x][y] = newCharacter

    if x < len(mapObj) - 1 and mapObj[x+1][y] == oldCharacter:
        floodFill(mapObj, x+1, y, oldCharacter, newCharacter) # call right
    if x > 0 and mapObj[x-1][y] == oldCharacter:
        floodFill(mapObj, x-1, y, oldCharacter, newCharacter) # call left
    if y < len(mapObj[x]) - 1 and mapObj[x][y+1] == oldCharacter:
        floodFill(mapObj, x, y+1, oldCharacter, newCharacter) # call down
    if y > 0 and mapObj[x][y-1] == oldCharacter:
        floodFill(mapObj, x, y-1, oldCharacter, newCharacter) # call up


def drawMap(mapObj, gameStateObj, goals, ai_mode):
    """Draws the map to a Surface object, including the player and
    stars. This function does not call pygame.display.update(), nor
    does it draw the "Level" and "Steps" text in the corner."""

    # mapSurf will be the single Surface object that the tiles are drawn
    # on, so that it is easy to position the entire map on the DISPLAYSURF
    # Surface object. First, the width and height must be calculated.
    col_num = len(mapObj[0])
    row_num = len(mapObj)
    mapSurfWidth = col_num * TILEWIDTH  
    mapSurfHeight = row_num * TILEHEIGHT 
    mapSurf = pygame.Surface((mapSurfWidth, mapSurfHeight))
    mapSurf.fill(BGCOLOR) # start with a blank color on the surface.

    # Draw the tile sprites onto this surface.
    for x in range(row_num):
        for y in range(col_num):
            spaceRect = pygame.Rect((y * TILEWIDTH, x * TILEHEIGHT, TILEWIDTH, TILEHEIGHT))
            if mapObj[x][y] in TILEMAPPING:
                baseTile = TILEMAPPING[mapObj[x][y]]
            elif mapObj[x][y] in OUTSIDEDECOMAPPING:
                baseTile = TILEMAPPING[' ']

            # First draw the base ground/wall tile.
            mapSurf.blit(baseTile, spaceRect)

            if mapObj[x][y] in OUTSIDEDECOMAPPING:
                # Draw any tree/rock decorations that are on this tile.
                mapSurf.blit(OUTSIDEDECOMAPPING[mapObj[x][y]], spaceRect)
            elif (x, y) in gameStateObj['stars']:
                if (x, y) in goals:
                    # A goal AND star are on this space, draw goal first.
                    mapSurf.blit(IMAGESDICT['covered goal'], spaceRect)
                # Then draw the star sprite.
                spaceRect = pygame.Rect((y * TILEWIDTH+5, x * TILEHEIGHT+5, TILEWIDTH, TILEHEIGHT))
                mapSurf.blit(IMAGESDICT['star'], spaceRect)
            elif (x, y) in goals:
                
                # Draw a goal without a star on it.
                mapSurf.blit(IMAGESDICT['uncovered goal'], spaceRect)

            # Last draw the player on the board.
            if (x, y) == gameStateObj['player']:
                # Note: The value "currentImage" refers to a key in "PLAYERIMAGES" which has the
                # specific player image we want to show.
                spaceRect = pygame.Rect((y * TILEWIDTH+5, x * TILEHEIGHT+5, TILEWIDTH, TILEHEIGHT))
                if ai_mode:
                    mapSurf.blit(ROBOTIMAGE, spaceRect)
                else:
                    mapSurf.blit(PLAYERIMAGES[currentImage], spaceRect)

    return mapSurf


def isLevelFinished(levelObj, gameStateObj):
    """Returns True if all the goals have stars in them."""
    for goal in levelObj['goals']:
        if goal not in gameStateObj['stars']:
            # Found a space with a goal but no star on it.
            return False
    return True


def terminate(process):
    # check if the ai_action_process is triggered or not
    if process is not None and process.is_alive():
        print("Terminating background process...")
        process.terminate()
        process.join()
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()