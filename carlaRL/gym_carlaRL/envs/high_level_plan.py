from enum import Enum
from collections import deque
import random
import time
import numpy as np
import carla
import sys
import xml.etree.ElementTree as ET
import networkx as nx
sys.path.append('C:/v-e2e-rl-ad/carlaRL/gym_carlaRL/envs/misc') # tweak to where you put carla

from .misc import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle
from .route_planner import compute_connection_original

class plan():
    def __init__(self, world):
        self.world = world
        self.map = self.world.get_map
        self.waypoints = self.map.generate_waypoints(2.0)
        self.nodes_edges

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

    def get_entrances_from_junctionID(self, junction_id):
        """
        Given a junction ID, find all entrance waypoints to junction.

        :param junction_id: string id of junction
        :return: list of all entrance locations to junction
        """
        list = []
        topology = self.map.get_topology()
        for wp in topology: #for each road segment, identify the start and end of road, and the road id
            start_waypoint = wp[0]
            end_waypoint = wp[1]
            if start_waypoint.is_junction: #if waypoint is in a junction
                if str(start_waypoint.junction_id) == junction_id:
                    list.append(start_waypoint.transform.location)    
            if end_waypoint.is_junction: #if waypoint is in a junction
                if str(end_waypoint.junction_id) == junction_id:
                    list.append(end_waypoint.transform.location)
  
        return list
    
    def find_intersection_waypoints_for_map(self):
        """
        Find all waypoints in every junction.

        :return: list of all waypoint locations in each junction on map
        """
        intersection_locations = [] #list of intersection coordinates

        for waypoint in  self.waypoints:
            if waypoint.is_junction: #if waypoint is in a junction
                intersection_locations.append(waypoint.transform.location) #get point in coordinate system for intersection location

        return intersection_locations

    def get_waypoints_for_junction(self, junction_id):
        """
        Given a specific junction ID, find all waypoints in junction.

        :param junction_id: string of specific junction ID
        :return: list of all specific junction waypoints in map
        """

        all_waypoints = self.map.generate_waypoints(2.0) #get list of all waypoints in map

        #filter waypoints that belong to specific junction id
        junction_waypoints = [wp for wp in all_waypoints if wp.is_junction and wp.get_junction().id == junction_id]
        return junction_waypoints

    def get_nodes_and_edges(self): #needs debugging
        """
        Get junctions and roads they connect to

        :return: dictionary of each junction with a list of its connected edges
        """
        topology = self.map.get_topology() #get all road segments in map
        junctions = set() #set to ensure uniqueness
        map_dict = {} #each junction ID has edges it connects to

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
                    if junctionID in map_dict:
                        map_dict[junctionID].add(road_id)
                    else:
                        map_dict[junctionID] = set()
                        map_dict[junctionID].add(road_id) 
            
            #get location of end of junction and get ID
            if end_waypoint.is_junction:
                junctionID = end_waypoint.get_junction().id
                if (end_waypoint_location, junctionID) not in junctions:
                    junctions.add((end_waypoint_location, junctionID)) 
                    if junctionID in map_dict:
                        map_dict[junctionID].add(road_id)
                    else:
                        map_dict[junctionID] = set()
                        map_dict[junctionID].add(road_id) 

        return map_dict


def find_intersections_directions_for_path(GRP, start, goal): #must debug
  """
  Given start and goal waypoints, find each intersection it must cross through and which direction to go at each 
  intersection.

  :param GRP: global route planner object
  :param start: starting location waypoint
  :param goal: goal location waypoint
  :return: list of [carla.Waypoint, RoadOption] to get from start to goal along path
  """
  intersections_directions = GRP.trace_route(start, goal) # get a list of [carla.Waypoint, RoadOption] to get from start to goal
  for pair in intersections_directions:
    if not pair[0].is_junction: #only put intersection locations and their road option in plan
      intersections_directions.remove(pair)
  return intersections_directions #list of intersections we need to get to and which direction to go when we get there

def find_all_nodes_and_edges(map):
  """
  Given a map, find all intersections and their connections to other intersections.

  :param map: world map object
  :return: dictionary of all connections for each intersection
  """
  intersection_locations = find_intersection_waypoints_for_map(map)
  node_edge_dict = {}
  for intersection in intersection_locations:
    waypoint = map.get_waypoint(intersection, project_to_road=True, lane_type=carla.LaneType.Any)
    
    # Initialize the list of connections for this junction
    connections = []

    # Get all junction waypoints connected to the current junction waypoint
    next_waypoints = waypoint.next_until_lane_end(2.0) + waypoint.previous_until_lane_start(2.0)
    for next_waypoint in next_waypoints:
      if next_waypoint.is_junction:
        next_loc = (next_waypoint.transform.location.x, next_waypoint.transform.location.y, next_waypoint.transform.location.z)
        if next_loc in intersection_locations and next_loc != intersection:
          connections.append(next_loc)

    node_edge_dict[intersection] = connections

  return node_edge_dict

def get_directed_edge(location, objective, map):
  """
  Given a map, a location, and another location, return the roadoption to get from location to 
  objective.

  :param intersection: any location object, but intended for intersections
  :param objective: any location object, but intended for other instersections that connect to first argument
  :param map: world map object
  :return: dictionary of all connections for each intersection
  """
  location_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
  objective_waypoint = map.get_waypoint(objective, project_to_road=True, lane_type=carla.LaneType.Any)

  direction = compute_connection_original(location_waypoint, objective_waypoint)

  return (direction, objective)

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
        color=carla.Color(r=0, g=0, b=255), life_time=60.0,
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


