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

sys.path.append('C:/v-e2e-rl-ad/carlaRL/gym_carlaRL/envs/route_planner')
from gym_carlaRL.envs.route_planner import find_intersections_for_map, debug_intersections, find_all_nodes_and_edges, find_intersections_directions_for_path


if __name__ == '__main__':
    # connect to the sim 
    client = carla.Client('localhost', 2000)

    world = client.get_world()

    spectator = world.get_spectator()

    # get map look at the map
    town_map = world.get_map()
    print("Map is loaded")

    #GRP = GlobalRoutePlanner(town_map, 1)
    print("global route planner created")

    intersections = find_intersections_for_map(town_map) #now have locations for intersections
    print("Got intersection locations")
    debug_intersections(world,intersections,spectator) #draw intersections on screen for 60 sec

    print("Testing connections")
    find_all_nodes_and_edges(town_map)

    print("Testing intersections and directions")

    #find_intersections_directions_for_path(GRP, start, goal)


