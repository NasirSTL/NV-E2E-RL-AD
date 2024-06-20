from enum import Enum
from collections import deque
import random
import time
import numpy as np
import carla
import sys
import xml.etree.ElementTree as ET
import networkx as nx
import math
sys.path.append('C:/carla/WindowsNoEditor/PythonAPI/carla/v-e2e-rl-ad/carlaRL/gym_carlaRL/envs/misc') 
sys.path.append('C:/carla/WindowsNoEditor/PythonAPI/carla/agents/navigation/global_route_planner') # tweak to where you put carla

from misc import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle
from route_planner import compute_connection_original
import agents
from agents.navigation.global_route_planner2 import GlobalRoutePlanner


class node():
    def __init__(self, junction_id, map):
      self.junction_id = junction_id
      self.map = map
      self.edges = set()

    def get_id(self):
       return self.junction_id

    def get_edges(self):
       return self.edges
    
    def get_all_waypoints(self):
      """
      Find all waypoints in junction.

      :return: list of all specific junction waypoints in map
      """

      all_waypoints = self.map.generate_waypoints(2.0) #get list of all waypoints in map

      #filter waypoints that belong to specific junction id
      junction_waypoints = []
      for wp in all_waypoints:
         if wp.is_junction and wp.get_junction().id == self.junction_id:
            junction_waypoints.append(wp)

      return junction_waypoints

    def get_entrances(self, map):
      """
      Using junction ID, find all entrance waypoints to junction.

      :return: list of all entrance locations to junction
      """
      list = []
      topology = map.get_topology()
      for wp in topology: #for each road segment, identify the start and end of road, and the road id
        start_waypoint = wp[0]
        end_waypoint = wp[1]
        if start_waypoint.is_junction: #if waypoint is in a junction
            if str(start_waypoint.junction_id) == self.junction_id:
                list.append(start_waypoint.transform.location)    
        if end_waypoint.is_junction: #if waypoint is in a junction
            if str(end_waypoint.junction_id) == self.junction_id:
                list.append(end_waypoint.transform.location)
  
      return list
    
    def get_location_from_junctionid():
      return
    
class plan():

  def __init__(self, world, start, goal):
     self.world = world
     self.map = self.world.get_map
     self.path = []
     self.goal = goal
     self.start = start

  def get_path(self, graph): #return nodes and edges to go through
     path_list = []
     nodes_in_map = graph.get_nodes_and_edges() #dictionary of junction id and connected edges
     sampling_resolution = 1
     grp = GlobalRoutePlanner(self.map, sampling_resolution, self.world)
     intersections_directions = grp.trace_route(self.start, self.goal) # get a list of [carla.Waypoint, RoadOption] to get from start to goal
     found_junction = True
     for pair in intersections_directions:
       if pair[0].is_junction and found_junction == False: 
         junction_node = node(pair[0].junction_id, self.map)
         path_list.append(junction_node.junction_id)
         found_junction = True
       if found_junction == True and pair[0].is_junction == False: #are now leaving junction
         path_list.append(pair[0].road_id)
         found_junction = False

     return path_list #list of intersections we need to get to and which road to go to from there


  def find_intersections_directions_for_path(self): #must debug
    """
    Find each intersection the car must cross through and which direction to go at each 
    intersection.

    :return: list of [carla.Waypoint, RoadOption] to get from start to goal along path
    """
    intersections_directions = self.global_route_planner.trace_route(self.start, self.goal) # get a list of [carla.Waypoint, RoadOption] to get from start to goal
    for pair in intersections_directions:
      if not pair[0].is_junction: #only put intersection locations and their road option in plan
        intersections_directions.remove(pair)
    return intersections_directions #list of intersections we need to get to and which direction to go when we get there


  def get_directed_edge(self):
    """
    Return the roadoption to get from location to 
    objective.

    :param intersection: any location object, but intended for intersections
    :param objective: any location object, but intended for other instersections that connect to first argument
    :param map: world map object
    :return: dictionary of all connections for each intersection
    """
    location_waypoint = self.map.get_waypoint(self.start, project_to_road=True, lane_type=carla.LaneType.Any)
    objective_waypoint = self.map.get_waypoint(self.goal, project_to_road=True, lane_type=carla.LaneType.Any)

    direction = compute_connection_original(location_waypoint, objective_waypoint)

    return (direction, self.goal)
  

class graph():
    def __init__(self, world, map):
        self.world = world
        self.map = map
        self.nodes = {}
        self.nodes_object = []


    def find_junctions_and_edges(self):
        """
        Get junctions and roads they connect to

        :return: list of nodes in map
        """
        topology = self.map.get_topology() #get all road segments in map
        junctions = set() #set to ensure uniqueness

        for waypoints in topology: #for each road segment, identify the start and end of road, and the road id
            start_waypoint = waypoints[0]
            end_waypoint = waypoints[1]
            road_id = start_waypoint.road_id
            start_waypoint_location = (start_waypoint.transform.location.x, start_waypoint.transform.location.y)
            end_waypoint_location = (end_waypoint.transform.location.x, end_waypoint.transform.location.y)

            #get location of start of junction and get junction ID
            if start_waypoint.is_junction:
                junctionID = start_waypoint.get_junction().id
                if (start_waypoint_location, junctionID) not in junctions:
                    junctions.add((start_waypoint_location, junctionID))
                    found = False
                    for junction_node in self.nodes_object:
                      if junction_node.junction_id == junctionID:
                        junction_node.edges.add(road_id)
                        found = True
                    if found == False:
                      junction_node = node(junctionID, self.map)
                      junction_node.edges.add(road_id)
                      self.nodes_object.append(junction_node)
            
            #get location of end of junction and get ID
            if end_waypoint.is_junction:
                junctionID = end_waypoint.get_junction().id
                if (end_waypoint_location, junctionID) not in junctions:
                    junctions.add((end_waypoint_location, junctionID)) 
                    found = False
                    for junction_node in self.nodes_object:
                      if junction_node.junction_id == junctionID:
                        junction_node.edges.add(road_id)
                        found = True
                    if found == False:
                      junction_node = node(junctionID, self.map)
                      junction_node.edges.add(road_id)
                      self.nodes_object.append(junction_node)

    def get_junction_ids(self):
        """
        Find all junction ID's in map.

        :return: list of all junction ID's in map
        """
        xodr_data = self.map.to_opendrive() #get map data
        root = ET.fromstring(xodr_data) #parse map data

        junctions = root.findall('junction') #get all junctions as list

        junction_ids = [junction.get('id') for junction in junctions] #get junction ID's as list
        return junction_ids
    
    def find_intersection_waypoints_for_map(self):
        """
        Find all waypoints in every junction.

        :return: list of all waypoint locations in each junction on map
        """
        intersection_locations = [] #list of intersection coordinates

        for waypoint in self.map.generate_waypoints(2.0):
            if waypoint.is_junction: #if waypoint is in a junction
                intersection_locations.append(waypoint.transform.location) #get point in coordinate system for intersection location

        return intersection_locations

    def get_nodes_and_edges(self): #needs debugging
        """
        Get junctions and roads they connect to

        :return: dictionary of each junction with a list of its connected edges
        """
        topology = self.map.get_topology() #get all road segments in map
        junctions = set() #set to ensure uniqueness

        for waypoints in topology: #for each road segment, identify the start and end of road, and the road id
            start_waypoint = waypoints[0]
            end_waypoint = waypoints[1]
            road_id = start_waypoint.road_id
            start_waypoint_location = (start_waypoint.transform.location.x, start_waypoint.transform.location.y)
            end_waypoint_location = (end_waypoint.transform.location.x, end_waypoint.transform.location.y)

            #get location of start of junction and get junction ID
            if start_waypoint.is_junction:
                junctionID = start_waypoint.get_junction().id
                if (start_waypoint_location, junctionID) not in junctions:
                    junctions.add((start_waypoint_location, junctionID))
                    if junctionID in self.nodes:
                        self.nodes[junctionID].add(road_id)
                    else:
                        self.nodes[junctionID] = set()
                        self.nodes[junctionID].add(road_id) 
            
            #get location of end of junction and get ID
            if end_waypoint.is_junction:
                junctionID = end_waypoint.get_junction().id
                if (end_waypoint_location, junctionID) not in junctions:
                    junctions.add((end_waypoint_location, junctionID)) 
                    if junctionID in self.nodes:
                        self.nodes[junctionID].add(road_id)
                    else:
                        self.nodes[junctionID] = set()
                        self.nodes[junctionID].add(road_id)

        return self.nodes

def debug_locations(world, locations, spectator):
  """
  Given a world, locations, and camera, draw all locations on map.

  :param world: world object
  :param location: list of coordinates
  :param spectator: spectator object
  """

  # draw all point in the sim for 60 seconds
  for wp in locations:
    world.debug.draw_string(wp, 'O', draw_shadow=False,
        color=carla.Color(r=0, g=0, b=255), life_time=2.0,
        persistent_lines=True)

  #move spectator for top down view to see all points 
  spectator_pos = carla.Transform(carla.Location(x=0,y=30,z=200),
                                carla.Rotation(pitch = -90, yaw = -90))
  spectator.set_transform(spectator_pos)

def get_junctions_from_edge(map_dict, road_id):  #needs debugging
  """
  Given junctions and the edges they connect to and a road id, identify which junctions the road connects to

  :param map_dict: dictionary of junction ID's and their edges
  :param road_id: road id of edge
  :return: dictionary of each junction with a list of its connected edges
  """
  connected_junctions = []
  for junction in map_dict:
    if map_dict[junction].contains(road_id):
      connected_junctions.append(junction)
  
  return connected_junctions   

