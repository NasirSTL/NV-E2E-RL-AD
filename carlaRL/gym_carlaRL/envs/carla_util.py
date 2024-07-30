import carla
from carla import ColorConverter
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import os
import cv2
import queue
import random
import numpy as np
import math

def carla_vec_to_np_array(vec):
    return np.array([vec.x,
                     vec.y,
                     vec.z])

class ImageProcessor():
    def __init__(self, controller_v=3, max_history_length=10, img_size=32, weight_img=True):
        self.controller_version = controller_v
        self.img_size = img_size
        self.weight_img = weight_img
        self.image_history = deque(maxlen=max_history_length)
        self.pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2)
        )

    def downsample_image(self, img):
        # Transform the image into tensor
        img = torch.tensor(img, dtype=torch.float32).cuda()
        img = torch.unsqueeze(img, dim=0)  # Add a batch dimension
        # Downsize the image
        img = F.interpolate(self.pooling(img), size=(32, 32), mode='bilinear', align_corners=False)
        # Convert the image back to numpy array, currently it is BxCxHxW, squeeze the batch dimension
        img = img.squeeze(0).cpu().numpy()

        return img

    def process_image(self, img):
        img_to_save = cv2.resize(img, (128,128))
        if self.img_size == 32:
            # Add a channel dimension to the image
            img = np.expand_dims(img, axis=0)
            # downsample the image
            img = self.downsample_image(img)
        elif self.img_size == 128:
            img = cv2.resize(img, (128,128))
            img = np.expand_dims(img, axis=0)
            
        if self.controller_version < 3:
            return img, img_to_save
        else:
            # Check if image history is full
            if len(self.image_history) == self.image_history.maxlen:
                self.image_history.popleft()
            self.image_history.append(img)

            # pad the list with the most recent image if the history is not full
            while len(self.image_history) < self.image_history.maxlen:
                self.image_history.append(self.image_history[-1])
        
            # sum the images in the history into a single image
            avg_img = np.zeros_like(self.image_history[0])
            for i, img in enumerate(self.image_history):
                if self.weight_img:
                    avg_img += (img * np.power(i+1, 3))  # Weight the image based on the time it was seen
                else:
                    avg_img += img
            
            min_val = np.min(avg_img)
            max_val = np.max(avg_img)
            
            if (max_val-min_val) == 0: #fail safe for nan in ImageProcessor
                print("Divide by 0 detected in ImageProcessor. Doing alternate normalization.")
                if min_val < 0 or max_val > 1:
                    img = np.where(avg_img <= 0, 0, 1)
                else:
                    img = avg_img
            else:
                # Normalize the image using min-max to the range [0, 1]
                img = (avg_img - np.min(avg_img)) / (np.max(avg_img) - np.min(avg_img))

            # squeeze the channel dimension to create image_to_save
            img_to_save = np.squeeze(img, axis=0)
            img_to_save = (img_to_save * 255).astype(np.uint8)

            return img, img_to_save
    
    def __len__(self):
        return len(self.image_history)

class DataSaver():
    def __init__(self, save_path='test'):
        self.save_path = os.path.join('gym_carlaRL/envs/data', save_path)
        mkdir_if_not_exist(self.save_path)
        self.counter = 0
        
    def save_image(self, image):
        path = os.path.join(self.save_path, f'raw_image')
        mkdir_if_not_exist(path)
        # resize image
        # image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        image_filename = os.path.join(path, f'{self.counter}.png')
        cv2.imwrite(image_filename, image)

    def save_third_pov_image(self, image):
        path = os.path.join(self.save_path, f'third_pov_image')
        mkdir_if_not_exist(path)
        image_filename = os.path.join(path, f'{self.counter}.png')
        cv2.imwrite(image_filename, image)

    def save_lane_image(self, image):
        path = os.path.join(self.save_path, f'lane_image')
        mkdir_if_not_exist(path)
        image_filename = os.path.join(path, f'{self.counter}.png')
        cv2.imwrite(image_filename, image)

    def save_metrics(self, metrics):
        metrics_filename = os.path.join(self.save_path, f'metrics.txt')
        with open(metrics_filename, 'a') as f:
            array_str = np.array2string(metrics, separator=', ', max_line_width=np.inf)
            # Remove the brackets [] from the start and end of the string representation
            array_str = array_str[1:-1]
            f.write(array_str)
            # Create a new line
            f.write('\n')

    def step(self):
        self.counter += 1


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            no_rendering_mode=False,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

# def carla_img_to_array(image):
#     # time_start = time.time()
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]
#     # print(f'took {time.time()-time_start:.2f}s to process the rgb image.')
#     return array

def create_carla_world(pygame, mapid, width, height):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    pygame.init()
    pygame.font.init()
    print(width, height)
    display = pygame.display.set_mode(
        (width, height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
        )
    display.fill((0,0,0))
    pygame.display.flip()
    font = get_font()
    clock = pygame.time.Clock()

    client.load_world('Town0' + mapid)
    world = client.get_world()
    return display, font, clock, world


def plot_map(m, mapid, vehicle = None):
    import matplotlib.pyplot as plt

    wp_list = m.generate_waypoints(2.0)
    loc_list = np.array(
        [carla_vec_to_np_array(wp.transform.location) for wp in wp_list]
    )
    plt.scatter(loc_list[:, 0], loc_list[:, 1])

    if vehicle != None:
        wp = m.get_waypoint(vehicle.get_transform().location)
        vehicle_loc = carla_vec_to_np_array(wp.transform.location)
        plt.scatter([vehicle_loc[0]], [vehicle_loc[1]])
        plt.title(f'Town0{mapid}')

    plt.show()


def draw_image(surface, image, blend=False):
    # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # array = np.reshape(array, (image.height, image.width, 4))
    # array = array[:, :, :3]
    # array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def get_curvature(polyline):
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    # print(curvature)
    return np.max(curvature)


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_random_spawn_point(CARLA_map):
    pose = random.choice(CARLA_map.get_spawn_points())
    return CARLA_map.get_waypoint(pose.location)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def find_weather_presets():
    """
    presets: ['ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNight', 
        'CloudyNoon', 'CloudySunset', 'Default', 'HardRainNight', 
        'HardRainNoon', 'HardRainSunset', 'MidRainSunset', 'MidRainyNight', 
        'MidRainyNoon', 'SoftRainNight', 'SoftRainNoon', 'SoftRainSunset', 
        'WetCloudyNight', 'WetCloudyNoon', 'WetCloudySunset', 'WetNight', 
        'WetNoon', 'WetSunset']
    
    return: [<Class Weather>, "Weather"] 
    E.g: [<Class ClearNight>, "ClearNight"]
    """
    import re
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    output = [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    return output

# -------------------------------------------------------------
# ---------------------Dynamic weather-------------------------
# -------------------------------------------------------------
def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        # Determine if it's day or night based on the altitude of the sun
        is_day = self.altitude > 0

        # Adjust the rate of change of _t based on day or night
        if is_day:
            delta_multiplier = 0.004  # Slower change during the day
        else:
            delta_multiplier = 0.012  # Faster change during the night

        self._t += delta_multiplier * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    # def tick(self, delta_seconds):
    #     self._t += 0.008 * delta_seconds
    #     self._t %= 2.0 * math.pi
    #     self.azimuth += 0.25 * delta_seconds
    #     self.azimuth %= 360.0
    #     self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        # Determine the storming condition based on the rain intensity
        is_storming = self.rain > 40.0  # Assuming storming if rain > 40.0, adjust as needed

        # Adjust the rate of change based on storming condition
        if is_storming:
            delta_multiplier = 2.0  # Faster change when storming
        else:
            delta_multiplier = 0.5  # Slower change when not storming

        delta = (delta_multiplier if self._increasing else -delta_multiplier) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    # def tick(self, delta_seconds):
    #     delta = (1.3 if self._increasing else -1.3) * delta_seconds
    #     self._t = clamp(delta + self._t, -250.0, 100.0)
    #     self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
    #     self.rain = clamp(self._t, 0.0, 80.0)
    #     delay = -10.0 if self._increasing else 90.0
    #     self.puddles = clamp(self._t + delay, 0.0, 85.0)
    #     self.wetness = clamp(self._t * 5, 0.0, 100.0)
    #     self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
    #     self.fog = clamp(self._t - 10, 0.0, 30.0)
    #     if self._t == -250.0:
    #         self._increasing = True
    #     if self._t == 100.0:
    #         self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)