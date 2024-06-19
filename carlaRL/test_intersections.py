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

sys.path.append('C:/v-e2e-rl-ad/carlaRL/gym_carlaRL/envs/high_level_plan')
import high_level_plan
from gym_carlaRL.envs.high_level_plan import find_intersection_waypoints_for_map, debug_locations, find_all_nodes_and_edges, find_intersections_directions_for_path
from gym_carlaRL.envs.high_level_plan import get_high_level_map, get_junction_ids, get_entrances_from_junctionID

if __name__ == '__main__':
    # connect to the sim 
    client = carla.Client('localhost', 2000)

    world = client.get_world()

    spectator = world.get_spectator()
    client.load_world('Town04')

    # get map look at the map
    town_map = world.get_map()

    plan = high_level_plan.plan(world)
    
    print("Map is loaded")

    """
     #GRP = GlobalRoutePlanner(town_map, 1)
    print("global route planner created")

    intersections = find_intersection_waypoints_for_map(town_map) #now have locations for intersections
    print("Got intersection locations")
    debug_locations(world,intersections,spectator) #draw intersections on screen for 60 sec

    print("Testing connections")
    find_all_nodes_and_edges(town_map)

    print("Testing intersections and directions")
    #find_intersections_directions_for_path(GRP, start, goal)
    """
    print("Testing nodes and edges")
    map_dict = plan.get_nodes_and_edges()
    print(map_dict)
    junction_ids = get_junction_ids(town_map)
    locations1 = [] 
    for id in junction_ids:
        locations = (get_entrances_from_junctionID(id, town_map, world, ))
        for l in locations:
            locations1.append(l)
    debug_locations(world,locations1,spectator)







