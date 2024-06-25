#all imports
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import time 
import cv2 #to work with images from cameras
import numpy as np 
import math
import sys
import random

sys.path.append('C:/carla/WindowsNoEditor/PythonAPI/carla') 
import carla #the sim library itself
#from agents.navigation.global_route_planner import GlobalRoutePlanner must debug

sys.path.append('C:/carla/WindowsNoEditor/PythonAPI/carla/v-e2e-rl-ad/carlaRL/gym_carlaRL/envs/high_level_plan')
#from high_level_plan import graph, plan
#from high_level_plan import *

if __name__ == '__main__':
    # connect to the sim 
    client = carla.Client('localhost', 2000)

    world = client.get_world()
    client.set_timeout(2000.0)
    spectator = world.get_spectator()
    client.load_world('Town05')




    """
     # get map look at the map
    town_map = world.get_map()

    all_waypoints = town_map.generate_waypoints(2.0)

    rand_start = random.choice(all_waypoints)
    rand_goal = random.choice(all_waypoints)
    
    start_loc = rand_start.transform.location
    goal_loc = rand_goal.transform.location


    world.debug.draw_string(start_loc, 'O', draw_shadow=False,color=carla.Color(r=0, g=0, b=255), life_time=60.0,
        persistent_lines=True)
    
    time.sleep(10)

    world.debug.draw_string(goal_loc, 'O', draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=60.0,
        persistent_lines=True)
    
    time.sleep(10)
    
    path = plan(world, start_loc, goal_loc)
    path_list = path.get_high_level_plan()


    print(path_list)
    """
    
    
    
    
    



