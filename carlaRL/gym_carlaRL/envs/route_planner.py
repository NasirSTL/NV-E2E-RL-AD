from enum import Enum
from collections import deque
import random
import time
import numpy as np
import carla
import sys
import xml.etree.ElementTree as ET
sys.path.append('C:/v-e2e-rl-ad/carlaRL/gym_carlaRL/envs/misc') # tweak to where you put carla

from .misc import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle

class RoadOption(Enum):
  """
  RoadOption represents the possible topological configurations when moving from a segment of lane to other.
  """
  VOID = -1
  LEFT = 1
  RIGHT = 2
  STRAIGHT = 3
  LANEFOLLOW = 4

class RoutePlanner():
  def __init__(self, vehicle, buffer_size):
    self._vehicle = vehicle
    self._world = self._vehicle.get_world()
    self._map = self._world.get_map()

    self._sampling_radius = 2
    self._min_distance = 4

    self._target_waypoint = None
    self._buffer_size = buffer_size
    self._waypoint_buffer = deque(maxlen=self._buffer_size)

    self._waypoints_queue = deque(maxlen=1000)
    self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
    self._waypoints_queue.append( (self._current_waypoint.next(self._sampling_radius)[0], 1))
    self._target_road_option = 1

    self._last_traffic_light = None
    self._proximity_threshold = 15.0

    self._compute_next_waypoints(k=500)

  def _compute_next_waypoints(self, k=1):
    """
    Add new waypoints to the trajectory queue.

    :param k: how many waypoints to compute
    :return:
    """
    available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
    k = min(available_entries, k)

    for _ in range(k):
      last_waypoint = self._waypoints_queue[-1][0]
      next_waypoints = list(last_waypoint.next(self._sampling_radius))

      if len(next_waypoints) == 0:
        break
      elif len(next_waypoints) == 1:
        # only one option available ==> lanefollowing
        next_waypoint = next_waypoints[0]
        road_option = compute_connection(last_waypoint, next_waypoint)
        if road_option != 1:
          break
      else:
        # random choice between the possible options
        road_options_list = retrieve_options(
          next_waypoints, last_waypoint)
        
        road_option = 0
        for opt in road_options_list:
          if opt == 1:
            road_option = 1
            next_waypoint = next_waypoints[road_options_list.index(road_option)]
            break
        if road_option == 0:
          break

      self._waypoints_queue.append((next_waypoint, road_option))


  def run_step(self):
    waypoints, not_straight = self._get_waypoints()
    # red_light, vehicle_front = self._get_hazard()
    # red_light = False
    return waypoints, not_straight

  def _get_waypoints(self):
    """
    Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
    follow the waypoints trajectory.

    :param debug: boolean flag to activate waypoints debugging
    :return:
    """

    # not enough waypoints in the horizon? => add more!
    if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.05):
      self._compute_next_waypoints(k=200)

    # Buffering the waypoints
    while len(self._waypoint_buffer)<self._buffer_size:
      if self._waypoints_queue:
        self._waypoint_buffer.append(self._waypoints_queue.popleft())
      else:
        break

    waypoints=[]
    not_straight = False

    for i, (waypoint, option) in enumerate(self._waypoint_buffer):
      if option == 1:
        waypoints.append((waypoint, option))
      else:
        not_straight = True
        break

    # current vehicle waypoint
    self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
    # target waypoint
    self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

    # purge the queue of obsolete waypoints
    vehicle_transform = self._vehicle.get_transform()
    num_waypoint_removed = 0

    for i, (waypoint, _) in enumerate(self._waypoint_buffer):
      d = distance_vehicle(waypoint, vehicle_transform)
    #   print(f"for {i} waypoint, its distance to vehicle: {d:.2f}")
      if d < self._min_distance:
        num_waypoint_removed += 1
      else:
        break
    
    # Only remove waypoints if necessary, and do it in one operation
    if num_waypoint_removed > 0:
      for _ in range(num_waypoint_removed):
        self._waypoint_buffer.popleft()

    # time_modify_buffer = time.time() - time_start
    # print(f"Time taken to modify buffer: {time_modify_buffer:.2f}s")

    return waypoints, not_straight

  def _get_hazard(self):
    # retrieve relevant elements for safe navigation, i.e.: traffic lights
    # and other vehicles
    actor_list = self._world.get_actors()
    vehicle_list = actor_list.filter("*vehicle*")
    lights_list = actor_list.filter("*traffic_light*")

    # check possible obstacles
    vehicle_state = self._is_vehicle_hazard(vehicle_list)

    # check for the state of the traffic lights
    light_state = self._is_light_red_us_style(lights_list)

    return light_state, vehicle_state

  def _is_vehicle_hazard(self, vehicle_list):
    """
    Check if a given vehicle is an obstacle in our way. To this end we take
    into account the road and lane the target vehicle is on and run a
    geometry test to check if the target vehicle is under a certain distance
    in front of our ego vehicle.

    WARNING: This method is an approximation that could fail for very large
     vehicles, which center is actually on a different lane but their
     extension falls within the ego vehicle lane.

    :param vehicle_list: list of potential obstacle to check
    :return: a tuple given by (bool_flag, vehicle), where
         - bool_flag is True if there is a vehicle ahead blocking us
           and False otherwise
         - vehicle is the blocker object itself
    """

    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    for target_vehicle in vehicle_list:
      # do not account for the ego vehicle
      if target_vehicle.id == self._vehicle.id:
        continue

      # if the object is not in our lane it's not an obstacle
      target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
      if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
              target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
        continue

      loc = target_vehicle.get_location()
      if is_within_distance_ahead(loc, ego_vehicle_location,
                    self._vehicle.get_transform().rotation.yaw,
                    self._proximity_threshold):
        return True

    return False

  def _is_light_red_us_style(self, lights_list):
    """
    This method is specialized to check US style traffic lights.

    :param lights_list: list containing TrafficLight objects
    :return: a tuple given by (bool_flag, traffic_light), where
         - bool_flag is True if there is a traffic light in RED
           affecting us and False otherwise
         - traffic_light is the object itself or None if there is no
           red traffic light affecting us
    """
    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    if ego_vehicle_waypoint.is_intersection:
      # It is too late. Do not block the intersection! Keep going!
      return False

    if self._target_waypoint is not None:
      if self._target_waypoint.is_intersection:
        potential_lights = []
        min_angle = 180.0
        sel_magnitude = 0.0
        sel_traffic_light = None
        for traffic_light in lights_list:
          loc = traffic_light.get_location()
          magnitude, angle = compute_magnitude_angle(loc,
                                 ego_vehicle_location,
                                 self._vehicle.get_transform().rotation.yaw)
          if magnitude < 80.0 and angle < min(25.0, min_angle):
            sel_magnitude = magnitude
            sel_traffic_light = traffic_light
            min_angle = angle

        if sel_traffic_light is not None:
          if self._last_traffic_light is None:
            self._last_traffic_light = sel_traffic_light

          if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
            return True
        else:
          self._last_traffic_light = None

    return False

def retrieve_options(list_waypoints, current_waypoint):
  """
  Compute the type of connection between the current active waypoint and the multiple waypoints present in
  list_waypoints. The result is encoded as a list of RoadOption enums.

  :param list_waypoints: list with the possible target waypoints in case of multiple options
  :param current_waypoint: current active waypoint
  :return: list of RoadOption enums representing the type of connection from the active waypoint to each
       candidate in list_waypoints
  """
  options = []
  for next_waypoint in list_waypoints:
    # this is needed because something we are linking to
    # the beggining of an intersection, therefore the
    # variation in angle is small
    next_next_waypoint = next_waypoint.next(3.0)[0]
    link = compute_connection(current_waypoint, next_next_waypoint)
    options.append(link)

  return options


def compute_connection(current_waypoint, next_waypoint, threshold=35):
  """
  Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
  (next_waypoint).

  :param current_waypoint: active waypoint
  :param next_waypoint: target waypoint
  :return: the type of topological connection encoded as a RoadOption enum:
       RoadOption.STRAIGHT
       RoadOption.LEFT
       RoadOption.RIGHT
  """
  n = next_waypoint.transform.rotation.yaw
  n = n % 360.0

  c = current_waypoint.transform.rotation.yaw
  c = c % 360.0

  diff_angle = (n - c) % 180.0
  if diff_angle < threshold or diff_angle > (180 - threshold):
    return 1
  elif diff_angle > 90.0:
    return 0
  else:
    return 0
  
def compute_connection_original(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

  
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

def find_intersections_for_map(map):
  """
  Given a map, find all intersections.

  :param map: world map object
  :return: list of all intersection locations in map
  """
  intersection_locations = [] #list of intersection coordinates
  waypoint_list = map.generate_waypoints(distance=2.0)  # Generate waypoints with a distance of 2 meters

  for waypoint in waypoint_list:
      if waypoint.is_junction: #if waypoint is in a junction
          intersection_locations.append(waypoint.transform.location) #get point in coordinate system for intersection location

  return intersection_locations

def find_all_nodes_and_edges(map):
  """
  Given a map, find all intersections and their connections to other intersections.

  :param map: world map object
  :return: dictionary of all connections for each intersection
  """
  intersection_locations = find_intersections_for_map(map)
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

def get_directed_edge(intersection, objective, map):
  """
  Given a map, an intersection location, and another location, return the roadoption to get from intersection to 
  objective.

  :param intersection: any location object, but intended for intersections
  :param objective: any location object, but intended for other instersections that connect to first argument
  :param map: world map object
  :return: dictionary of all connections for each intersection
  """
  intersection_waypoint = map.get_waypoint(intersection, project_to_road=True, lane_type=carla.LaneType.Any)
  objective_waypoint = map.get_waypoint(objective, project_to_road=True, lane_type=carla.LaneType.Any)

  direction = compute_connection_original(intersection_waypoint, objective_waypoint)

  return (direction, objective)

def debug_intersections(world, locations, spectator):
  """
  Given a world, locations, and camera, draw all locations on map.

  :param world: world object
  :param location: list of coordinates
  :param spectator: spectator object
  """

  # draw all point in the sim for 60 seconds
  for wp in locations:
    world.debug.draw_string(wp, 'O', draw_shadow=False,
        color=carla.Color(r=0, g=0, b=255), life_time=20.0,
        persistent_lines=True)

  #move spectator for top down view to see all points 
  spectator_pos = carla.Transform(carla.Location(x=0,y=30,z=200),
                                carla.Rotation(pitch = -90, yaw = -90))
  spectator.set_transform(spectator_pos)

def get_junction_ids(map):
  """
  Given a map object, find all junction ID's in map.

  :param map: map object
  :return: list of all junction ID's in map
  """
  xodr_data = map.to_opendrive() #get map data
  root = ET.fromstring(xodr_data) #parse map data

  junctions = root.findall('junction') #get all junctions as list

  junction_ids = [junction.get('id') for junction in junctions] #get junction ID's as list
  return junction_ids


def get_waypoints_in_junction(map, junction_id):
  """
  Given a map object and specific junction ID, find all junction ID's in map.

  :param map: map object
  :param junction_id: string of specific junction ID
  :return: list of all specifci junction waypoints in map
  """

  all_waypoints = map.generate_waypoints(2.0) #get list of all waypoints in map

  #filter waypoints that belong to specific junction id
  junction_waypoints = [wp for wp in all_waypoints if wp.is_junction and wp.get_junction().id == junction_id]
  return junction_waypoints



