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
#from high_level_plan import graph, plan
from agents.navigation.global_route_planner2 import GlobalRoutePlanner, RoadOption


  
class _plan():

  def __init__(self, world, start, goal):
     self.world = world
     self.map = self.world.get_map()
     self.path = []
     self.goal = goal
     self.start = start

  def get_high_level_plan2(self):
    sampling_resolution = 1
    grp = GlobalRoutePlanner(self.map, sampling_resolution)
    route = grp.trace_route(self.start, self.goal) # get a list of [carla.Waypoint, RoadOption] to get from start to goal
    high_level_plan = []
    current_command = route[0][1]
    high_level_plan.append([route[0][0].transform.location, RoadOption.LANEFOLLOW])
    j=0

    for i in range(len(route)):
      waypoint, command = route[i]
      prev_loc, prev_command = high_level_plan[j-1]
    
      if command != RoadOption.CHANGELANELEFT and command != RoadOption.CHANGELANERIGHT:
        if prev_loc.distance(waypoint.transform.location) < 1:
         high_level_plan.pop(j-1)
         high_level_plan.append([waypoint.transform.location, command])

        else:
            if waypoint.is_junction and command == RoadOption.LANEFOLLOW and current_command != RoadOption.STRAIGHT:
             high_level_plan.append([waypoint.transform.location, RoadOption.STRAIGHT])
             current_command = RoadOption.STRAIGHT
             j = j+1
      
            elif current_command != command:
             if command == RoadOption.CHANGELANERIGHT or command == RoadOption.CHANGELANELEFT:
                high_level_plan.append([waypoint.transform.location, RoadOption.LANEFOLLOW])
                current_command = RoadOption.LANEFOLLOW
                j = j+1
             else:
                high_level_plan.append([waypoint.transform.location, command])
                current_command = command
                j = j+1

    high_level_plan.append([self.goal, "STOP"])

    return high_level_plan


if __name__ == '__main__':
    # connect to the sim 
    client = carla.Client('localhost', 2000)

    world = client.get_world()
    client.set_timeout(2000.0)
    spectator = world.get_spectator()
    client.load_world('Town05')


     # get map look at the map
    town_map = world.get_map()

    all_waypoints = town_map.generate_waypoints(2.0)

    rand_start = random.choice(all_waypoints)
    rand_goal = random.choice(all_waypoints)
    
    start_loc = rand_start.transform.location
    goal_loc = rand_goal.transform.location


    world.debug.draw_string(start_loc, 'O', draw_shadow=False,color=carla.Color(r=0, g=0, b=255), life_time=200.0,
        persistent_lines=True)
    
    time.sleep(10)

    world.debug.draw_string(goal_loc, 'O', draw_shadow=False,color=carla.Color(r=0, g=255, b=0), life_time=200.0,
        persistent_lines=True)
    
    time.sleep(10)
    
    path = _plan(world, start_loc, goal_loc)
    path_list = path.get_high_level_plan2() #list of [location, roadoption]
    i = 0
    for things in path_list:
        loc = things[0]
        option = things[1]
        world.debug.draw_string(loc, str(i), draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=100.0,
        persistent_lines=True)
        print(str(i), option)
        time.sleep(1)
        i = i+1

    
    
    
    
    



