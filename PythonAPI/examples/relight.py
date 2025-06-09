import sys
sys.path.append('/home/lewa/Documents/drive/CarlaUE5/PythonAPI/examples')  # 替换为实际路径

import carla
import random
import time
import numpy as np
import cv2
import os
from collections import deque
import argparse
from tqdm import tqdm

# Add imports for intelligent agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent
AGENTS_AVAILABLE = True

def wait(world, frames=100):
    for i in range(0, frames):
        world.tick()

class CarlaRelightSimulator:
    def __init__(self, host='127.0.0.1', port=2000, output_dir='./carla_relight_data'):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = None
        self.vehicle = None
        self.camera_rgb = None
        self.npc_vehicles = []  # Store NPC vehicles
        self.trajectory_data = []  # Store trajectory for replay
        self.output_dir = output_dir
        self.scene_idx = 0
        self.image_count = 0
        
        # Add agent system
        self.agent = None
        self.agent_type = 'behavior'  # Options: 'behavior', 'basic', 'constant', 'autopilot'
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhanced lighting configurations with complete weather parameters
        self.lighting_configs = [
            {
                'sun_altitude_angle': 40, 'cloudiness': 100, 'precipitation': 60, 'sun_azimuth_angle': 0,
                'precipitation_deposits': 85, 'wind_intensity': 90, 'fog_density': 20, 'fog_distance': 50, 'wetness': 80,
                'name': 'heavy_rain'
            },
            # {
            #     'sun_altitude_angle': 50, 'cloudiness': 90, 'precipitation': 30, 'sun_azimuth_angle': 315,
            #     'precipitation_deposits': 40, 'wind_intensity': 45, 'fog_density': 10, 'fog_distance': 100, 'wetness': 60,
            #     'name': 'light_rain'
            # },
            {
                'sun_altitude_angle': 90, 'cloudiness': 0, 'precipitation': 0, 'sun_azimuth_angle': 0,
                'precipitation_deposits': 0, 'wind_intensity': 10, 'fog_density': 0, 'fog_distance': 200, 'wetness': 0,
                'name': 'noon_clear'
            },
            {
                'sun_altitude_angle': 75, 'cloudiness': 20, 'precipitation': 0, 'sun_azimuth_angle': 45,
                'precipitation_deposits': 0, 'wind_intensity': 25, 'fog_density': 5, 'fog_distance': 150, 'wetness': 10,
                'name': 'morning_light_clouds'
            },
            {
                'sun_altitude_angle': 45, 'cloudiness': 0, 'precipitation': 0, 'sun_azimuth_angle': 90,
                'precipitation_deposits': 0, 'wind_intensity': 15, 'fog_density': 0, 'fog_distance': 200, 'wetness': 0,
                'name': 'morning_clear'
            },
            {
                'sun_altitude_angle': 30, 'cloudiness': 40, 'precipitation': 0, 'sun_azimuth_angle': 135,
                'precipitation_deposits': 0, 'wind_intensity': 35, 'fog_density': 8, 'fog_distance': 120, 'wetness': 20,
                'name': 'afternoon_cloudy'
            },
            {
                'sun_altitude_angle': 15, 'cloudiness': 0, 'precipitation': 0, 'sun_azimuth_angle': 180,
                'precipitation_deposits': 0, 'wind_intensity': 20, 'fog_density': 2, 'fog_distance': 180, 'wetness': 5,
                'name': 'sunset_clear'
            },
            {
                'sun_altitude_angle': 10, 'cloudiness': 60, 'precipitation': 0, 'sun_azimuth_angle': 225,
                'precipitation_deposits': 0, 'wind_intensity': 50, 'fog_density': 15, 'fog_distance': 80, 'wetness': 30,
                'name': 'sunset_cloudy'
            },
            # {
            #     'sun_altitude_angle': 70, 'cloudiness': 80, 'precipitation': 0, 'sun_azimuth_angle': 270,
            #     'precipitation_deposits': 0, 'wind_intensity': 40, 'fog_density': 12, 'fog_distance': 100, 'wetness': 25,
            #     'name': 'overcast'
            # },
            {
                'sun_altitude_angle': 85, 'cloudiness': 10, 'precipitation': 0, 'sun_azimuth_angle': 180,
                'precipitation_deposits': 0, 'wind_intensity': 5, 'fog_density': 0, 'fog_distance': 200, 'wetness': 0,
                'name': 'bright_sun'
            },
            # {
            #     'sun_altitude_angle': 60, 'cloudiness': 95, 'precipitation': 0, 'sun_azimuth_angle': 90,
            #     'precipitation_deposits': 0, 'wind_intensity': 20, 'fog_density': 80, 'fog_distance': 20, 'wetness': 40,
            #     'name': 'heavy_fog'
            # },
            # {
            #     'sun_altitude_angle': 35, 'cloudiness': 70, 'precipitation': 5, 'sun_azimuth_angle': 120,
            #     'precipitation_deposits': 10, 'wind_intensity': 15, 'fog_density': 50, 'fog_distance': 40, 'wetness': 35,
            #     'name': 'misty_morning'
            # },
            {
                'sun_altitude_angle': 25, 'cloudiness': 100, 'precipitation': 80, 'sun_azimuth_angle': 200,
                'precipitation_deposits': 90, 'wind_intensity': 95, 'fog_density': 30, 'fog_distance': 30, 'wetness': 95,
                'name': 'thunderstorm'
            },
            {
                'sun_altitude_angle': 55, 'cloudiness': 40, 'precipitation': 0, 'sun_azimuth_angle': 75,
                'precipitation_deposits': 70, 'wind_intensity': 25, 'fog_density': 5, 'fog_distance': 160, 'wetness': 85,
                'name': 'wet_roads_after_rain'
            },
            {
                'sun_altitude_angle': -20, 'cloudiness': 30, 'precipitation': 0, 'sun_azimuth_angle': 250,
                'precipitation_deposits': 0, 'wind_intensity': 80, 'fog_density': 3, 'fog_distance': 180, 'wetness': 15,
                'name': 'windy_evening'
            },
            {
                'sun_altitude_angle': -10, 'cloudiness': 10, 'precipitation': 0, 'sun_azimuth_angle': 0,
                'precipitation_deposits': 0, 'wind_intensity': 15, 'fog_density': 0, 'fog_distance': 200, 'wetness': 0,
                'name': 'clear_night'
            },
            {
                'sun_altitude_angle': -15, 'cloudiness': 70, 'precipitation': 0, 'sun_azimuth_angle': 45,
                'precipitation_deposits': 0, 'wind_intensity': 35, 'fog_density': 20, 'fog_distance': 80, 'wetness': 10,
                'name': 'cloudy_night'
            },
            {
                'sun_altitude_angle': -20, 'cloudiness': 90, 'precipitation': 40, 'sun_azimuth_angle': 90,
                'precipitation_deposits': 60, 'wind_intensity': 70, 'fog_density': 35, 'fog_distance': 50, 'wetness': 75,
                'name': 'rainy_night'
            },
            {
                'sun_altitude_angle': -25, 'cloudiness': 80, 'precipitation': 0, 'sun_azimuth_angle': 135,
                'precipitation_deposits': 0, 'wind_intensity': 25, 'fog_density': 60, 'fog_distance': 30, 'wetness': 20,
                'name': 'foggy_night'
            }
        ]

        self.available_maps = [
            'Town10HD', # High detail map
            'Town01',
            'Town02', 
            'Town03',
            'Town04',
            'Town05',
            'Town06',
            'Town07',
        ]

        
        # Simplified random scene generation - no complex strategies needed
        self.num_scenes_to_generate = 10  # Can easily generate 50+ scenes
    def setup_spectator(self, mode='behind_vehicle'):
        """Setup spectator view based on vehicle position"""
        # if not self.vehicle or not self.vehicle.is_alive:
        #     print("Vehicle not available for spectator setup")
        #     return False
        
        vehicle_transform = self.vehicle.get_transform()
        spectator = self.world.get_spectator()
        
        # Position spectator behind the vehicle looking forward
        spectator_transform = carla.Transform(
            vehicle_transform.location, 
            vehicle_transform.rotation
        )
        vehicle_bbox = self.vehicle.bounding_box
        vehicle_length = vehicle_bbox.extent.x 
        camera_x = vehicle_transform.get_forward_vector() * vehicle_length

        spectator_transform.location += camera_x 
        spectator_transform.location.z += 2
        # spectator_transform.rotation.yaw += config['yaw_offset']

        spectator = self.world.get_spectator()
        spectator.set_transform(spectator_transform)
        # wait(self.world)
        # breakpoint()
        self.current_spectator_mode = mode
        # print(f"Spectator set to {mode} mode")
        return True        
    


    def reset_world(self):
        """Reset world by destroying all vehicles and actors"""
        print("Resetting world - destroying all vehicles and actors...")
        
        # Get all actors in the world
        actors = self.world.get_actors()
        
        # Filter vehicles (including NPCs and main vehicle)
        vehicles = actors.filter('vehicle.*')
        
        # Destroy all vehicles
        destroyed_count = 0
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
                destroyed_count += 1
        
        # Also destroy sensors if any exist
        sensors = actors.filter('sensor.*')
        for sensor in sensors:
            if sensor.is_alive:
                sensor.destroy()
        
        # Clear our internal references
        self.vehicle = None
        self.camera_rgb = None
        self.npc_vehicles.clear()
        self.agent = None
        self.trajectory_data.clear()
        
        # Wait for destruction to complete
        if self.world.get_settings().synchronous_mode:
            for _ in range(10):
                self.world.tick()
        else:
            time.sleep(0.5)
        
        print(f"World reset completed - destroyed {destroyed_count} vehicles and {len(sensors)} sensors")
        return True
    

    def set_synchronous_mode(self, synchronous=True, fixed_delta_seconds=0.05):
        """Set synchronous mode according to CARLA best practices"""
        if self.world is None:
            print("World not initialized, cannot set synchronous mode")
            return False
        
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous
        
        if synchronous:
            # Use fixed time step for synchronous mode as recommended
            settings.fixed_delta_seconds = fixed_delta_seconds
            # Configure physics substepping for better precision
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
        else:
            settings.fixed_delta_seconds = None
        
        self.world.apply_settings(settings)
        
        if synchronous:
            print(f"Synchronous mode enabled with fixed delta: {fixed_delta_seconds}s")
            print(f"Physics substepping: {settings.max_substeps} substeps, max delta: {settings.max_substep_delta_time}s")
        else:
            print("Asynchronous mode enabled")
        
        return True

    def record_scene_trajectory(self, duration_seconds=75):
        """Record trajectory of main vehicle and NPCs for consistent replay"""
        print(f"Recording scene trajectory for {duration_seconds} seconds...")
        
        self.trajectory_data = []
        self.npc_trajectory_data = []
        
        # Setup intelligent agent for main vehicle
        agent_success = self.setup_intelligent_agent(self.agent_type)
        
        # Setup NPC vehicles with varied behaviors
        self.setup_npc_agents()
        
        # Calculate number of frames to record
        if self.world.get_settings().synchronous_mode:
            frames_to_record = int(duration_seconds / 0.05)  # 0.05s per tick
        else:
            frames_to_record = int(duration_seconds * 20)  # Approximate 20 FPS
        
        start_time = time.time()
        
        # Add progress bar for trajectory recording
        with tqdm(total=frames_to_record, desc="Recording trajectory", unit="frames") as pbar:
            for frame_idx in range(frames_to_record):
                # Update vehicle control using agent
                if agent_success and self.agent:
                    self.update_agent_control()
                
                # Update NPC control
                self.update_npc_control()

                self.setup_spectator()
                
                # Record main vehicle transform
                vehicle_transform = self.vehicle.get_transform()
                vehicle_velocity = self.vehicle.get_velocity()
                
                frame_data = {
                    'frame': frame_idx,
                    'vehicle_transform': {
                        'location': [vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.location.z],
                        'rotation': [vehicle_transform.rotation.pitch, vehicle_transform.rotation.yaw, vehicle_transform.rotation.roll]
                    },
                    'vehicle_velocity': [vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z]
                }
                
                # Record NPC transforms
                npc_transforms = []
                for i, npc in enumerate(self.npc_vehicles):
                    if npc.is_alive:
                        npc_transform = npc.get_transform()
                        npc_velocity = npc.get_velocity()
                        npc_data = {
                            'npc_id': i,
                            'transform': {
                                'location': [npc_transform.location.x, npc_transform.location.y, npc_transform.location.z],
                                'rotation': [npc_transform.rotation.pitch, npc_transform.rotation.yaw, npc_transform.rotation.roll]
                            },
                            'velocity': [npc_velocity.x, npc_velocity.y, npc_velocity.z]
                        }
                        npc_transforms.append(npc_data)
                
                frame_data['npc_transforms'] = npc_transforms
                self.trajectory_data.append(frame_data)
                
                # Update progress bar
                pbar.update(1)
                if frame_idx % 50 == 0:  # Update description every 50 frames
                    elapsed = time.time() - start_time
                    fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                    vehicle_speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2) * 3.6
                    pbar.set_postfix({
                        'FPS': f'{fps:.1f}', 
                        'NPCs': len(npc_transforms),
                        'Speed': f'{vehicle_speed:.1f}km/h'
                    })
                
                # Advance simulation
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    time.sleep(0.05)
        
        print(f"Trajectory recording completed: {len(self.trajectory_data)} frames recorded")
        return len(self.trajectory_data) > 0

    def replay_scene_trajectory(self, lighting_config, images_per_scene=5, map_name=""):
        """Replay recorded trajectory with specific lighting conditions"""
        if not self.trajectory_data:
            print("No trajectory data available for replay!")
            return False
        
        print(f"Replaying scene with lighting: {lighting_config['name']}")
        
        # Apply lighting conditions first
        self.setup_weather(lighting_config)
        
        # Additional stabilization for weather
        stabilization_ticks = 15 if lighting_config.get('fog_density', 0) > 50 else 10
        for _ in range(stabilization_ticks):
            self.world.tick()
        
        # Calculate which frames to capture images from
        total_frames = len(self.trajectory_data)
        if images_per_scene <= 1:
            capture_frames = [total_frames // 2]  # Middle frame
        else:
            # Evenly distribute capture frames
            frame_step = max(1, total_frames // images_per_scene)
            capture_frames = [i * frame_step for i in range(images_per_scene)]
            # Ensure we don't exceed available frames
            capture_frames = [min(f, total_frames - 1) for f in capture_frames]
        
        print(f"  Will capture images at frames: {capture_frames[:10]}... (total {len(capture_frames)})")
        
        # Disable autopilot for precise positioning
        self.vehicle.set_autopilot(False)
        for npc in self.npc_vehicles:
            if npc.is_alive:
                npc.set_autopilot(False)
        
        # Stabilization after disabling autopilot
        for _ in range(5):
            self.world.tick()
        
        captured_count = 0
        
        # Replay trajectory with progress bar
        with tqdm(total=len(self.trajectory_data), desc=f"Replaying {lighting_config['name']}", unit="frames") as pbar:
            for frame_idx, frame_data in enumerate(self.trajectory_data):
                # Set main vehicle position and velocity
                vehicle_loc = frame_data['vehicle_transform']['location']
                vehicle_rot = frame_data['vehicle_transform']['rotation']
                vehicle_vel = frame_data['vehicle_velocity']
                
                new_transform = carla.Transform(
                    carla.Location(x=vehicle_loc[0], y=vehicle_loc[1], z=vehicle_loc[2]),
                    carla.Rotation(pitch=vehicle_rot[0], yaw=vehicle_rot[1], roll=vehicle_rot[2])
                )
                
                self.vehicle.set_transform(new_transform)
                self.vehicle.set_target_velocity(carla.Vector3D(x=vehicle_vel[0], y=vehicle_vel[1], z=vehicle_vel[2]))
                
                # Set NPC positions and velocities
                for npc_data in frame_data['npc_transforms']:
                    npc_id = npc_data['npc_id']
                    if npc_id < len(self.npc_vehicles) and self.npc_vehicles[npc_id].is_alive:
                        npc_loc = npc_data['transform']['location']
                        npc_rot = npc_data['transform']['rotation']
                        npc_vel = npc_data['velocity']
                        
                        npc_transform = carla.Transform(
                            carla.Location(x=npc_loc[0], y=npc_loc[1], z=npc_loc[2]),
                            carla.Rotation(pitch=npc_rot[0], yaw=npc_rot[1], roll=npc_rot[2])
                        )
                        
                        self.npc_vehicles[npc_id].set_transform(npc_transform)
                        self.npc_vehicles[npc_id].set_target_velocity(carla.Vector3D(x=npc_vel[0], y=npc_vel[1], z=npc_vel[2]))
                
                # Advance simulation to apply transforms
                self.world.tick()
                
                # Capture image if this is a designated capture frame
                if frame_idx in capture_frames:
                    image_idx = capture_frames.index(frame_idx)
                    success = self.capture_single_image(lighting_config['name'], f"replay_scene_{self.scene_idx}", image_idx, map_name)
                    if success:
                        captured_count += 1
                        pbar.set_postfix({'Captured': f'{captured_count}/{len(capture_frames)}'})
                
                # Update progress bar
                pbar.update(1)
        
        print(f"  Replay completed: {captured_count}/{len(capture_frames)} images captured")
        return captured_count > 0

    def set_consistent_viewpoint(self, base_transform, img_idx, total_images):
        """Set consistent viewpoint variations for relight dataset"""
        if total_images == 1:
            # Single image: use exact base transform
            self.vehicle.set_transform(base_transform)
        else:
            # Multiple images: use deterministic variations based on index
            # This ensures same variations across all lighting conditions
            angle_step = 360.0 / total_images
            current_angle = img_idx * angle_step
            
            # Small, consistent positional variations
            offset_x = 0.5 * np.cos(np.radians(current_angle))
            offset_y = 0.5 * np.sin(np.radians(current_angle))
            
            new_location = carla.Location(
                x=base_transform.location.x + offset_x,
                y=base_transform.location.y + offset_y,
                z=base_transform.location.z
            )
            
            # Small, consistent rotational variations
            yaw_offset = current_angle * 0.1  # Small rotation factor
            
            new_rotation = carla.Rotation(
                pitch=base_transform.rotation.pitch,
                yaw=base_transform.rotation.yaw + yaw_offset,
                roll=base_transform.rotation.roll
            )
            
            new_transform = carla.Transform(new_location, new_rotation)
            self.vehicle.set_transform(new_transform)

    def setup_weather(self, config):
        """Apply weather conditions with enhanced synchronization for consistent dataset"""
        weather = carla.WeatherParameters(
            cloudiness=float(config['cloudiness']),
            precipitation=float(config['precipitation']),
            sun_altitude_angle=float(config['sun_altitude_angle']),
            sun_azimuth_angle=float(config['sun_azimuth_angle']),
            precipitation_deposits=float(config.get('precipitation_deposits', 0.0)),
            wind_intensity=float(config.get('wind_intensity', 0.0)),
            fog_density=float(config.get('fog_density', 0.0)),
            fog_distance=float(config.get('fog_distance', 200.0)),
            wetness=float(config.get('wetness', 0.0))
        )
        
        # Apply weather settings
        self.world.set_weather(weather)
        print(f"Applying weather: {config['name']}")
        print(f"  - Sun altitude: {config['sun_altitude_angle']}°, azimuth: {config['sun_azimuth_angle']}°")
        print(f"  - Cloudiness: {config['cloudiness']}%, precipitation: {config['precipitation']}%")
        print(f"  - Wind: {config.get('wind_intensity', 0)}%, fog: {config.get('fog_density', 0)}%")
        print(f"  - Wetness: {config.get('wetness', 0)}%, deposits: {config.get('precipitation_deposits', 0)}%")
        
        # Enhanced synchronization for consistent weather application
        if self.world.get_settings().synchronous_mode:
            max_sync_attempts = 40  # Increased for better consistency
            sync_tolerance = 1.5    # Tighter tolerance for dataset consistency
            
            for attempt in range(max_sync_attempts):
                self.world.tick()
                
                # Check weather synchronization more frequently
                if attempt % 3 == 2:  # Check every 3 ticks
                    current_weather = self.world.get_weather()
                    
                    # Verify critical weather parameters for relight consistency
                    sun_alt_diff = abs(current_weather.sun_altitude_angle - config['sun_altitude_angle'])
                    sun_azm_diff = abs(current_weather.sun_azimuth_angle - config['sun_azimuth_angle'])
                    cloudiness_diff = abs(current_weather.cloudiness - config['cloudiness'])
                    
                    if (sun_alt_diff < sync_tolerance and 
                        sun_azm_diff < sync_tolerance and 
                        cloudiness_diff < sync_tolerance):
                        print(f"Weather synchronized after {attempt+1} ticks")
                        break
                        
                    if attempt == max_sync_attempts - 1:
                        print(f"Warning: Weather may not be fully synchronized after {max_sync_attempts} ticks")
                        # Force additional stabilization
                        for _ in range(10):
                            self.world.tick()
        else:
            # In asynchronous mode, wait longer for stable weather
            wait_time = 4.0 if config.get('fog_density', 0) > 30 or config.get('precipitation', 0) > 50 else 3.0
            time.sleep(wait_time)
        
        # Verify weather application for dataset consistency
        self._verify_weather_application(config)

    def _verify_weather_application(self, target_config):
        """Verify that weather parameters were applied correctly"""
        current_weather = self.world.get_weather()
        
        print(f"Weather verification for {target_config['name']}:")
        
        # Define parameters to verify
        params_to_verify = [
            ('sun_altitude_angle', 'Sun altitude'),
            ('sun_azimuth_angle', 'Sun azimuth'),
            ('cloudiness', 'Cloudiness'),
            ('precipitation', 'Precipitation'),
            ('wind_intensity', 'Wind intensity'),
            ('fog_density', 'Fog density'),
            ('wetness', 'Wetness'),
            ('precipitation_deposits', 'Precipitation deposits')
        ]
        
        verification_failed = False
        tolerance = 3.0  # Allow some tolerance for weather parameters
        
        for param_name, display_name in params_to_verify:
            if param_name in target_config:
                target_value = target_config[param_name]
                current_value = getattr(current_weather, param_name)
                difference = abs(current_value - target_value)
                
                status = "✓" if difference <= tolerance else "✗"
                print(f"  {status} {display_name}: {current_value:.1f} (target: {target_value})")
                
                if difference > tolerance:
                    verification_failed = True
        
        if verification_failed:
            print(f"Warning: Some weather parameters for '{target_config['name']}' may not have applied correctly")

    def setup_intelligent_agent(self, agent_type='behavior'):
        """Setup intelligent agent for better vehicle control with improved target selection"""
        if not AGENTS_AVAILABLE or not self.vehicle:
            print("Using basic autopilot - agents not available")
            self.vehicle.set_autopilot(True)
            return False
        
        current_location = self.vehicle.get_location()
        spawn_points = self.world.get_map().get_spawn_points()
        
        if not spawn_points:
            print("No spawn points available for destination")
            self.vehicle.set_autopilot(True)
            return False
        
        # Find a suitable destination that is far enough from current position
        min_distance = 100.0  # Minimum 100 meters away
        suitable_destinations = []
        
        for spawn_point in spawn_points:
            distance = current_location.distance(spawn_point.location)
            if distance >= min_distance:
                suitable_destinations.append((spawn_point, distance))
        
        if not suitable_destinations:
            print(f"No destinations found beyond {min_distance}m, using all spawn points")
            suitable_destinations = [(sp, current_location.distance(sp.location)) for sp in spawn_points]
        
        # Sort by distance and select from the farther ones for longer driving
        suitable_destinations.sort(key=lambda x: x[1], reverse=True)
        top_destinations = suitable_destinations[:max(1, len(suitable_destinations)//3)]
        selected_destination, selected_distance = random.choice(top_destinations)
        
        print(f"Selected destination {selected_distance:.1f}m away")
        
        # Initialize the agent based on type
        agent_initialized = False
        
        if agent_type == 'behavior':
            self.agent = BehaviorAgent(self.vehicle, behavior='normal')
            print("Using BehaviorAgent with normal behavior")
            agent_initialized = True
        elif agent_type == 'basic':
            self.agent = BasicAgent(self.vehicle, target_speed=30)
            self.agent.follow_speed_limits(True)
            print("Using BasicAgent with target speed 30 km/h")
            agent_initialized = True
        elif agent_type == 'constant':
            self.agent = ConstantVelocityAgent(self.vehicle, target_speed=25)
            # Ensure vehicle is on ground
            ground_loc = self.world.ground_projection(self.vehicle.get_location(), 5)
            if ground_loc:
                self.vehicle.set_location(ground_loc.location + carla.Location(z=0.01))
            self.agent.follow_speed_limits(True)
            print("Using ConstantVelocityAgent with target speed 25 km/h")
            agent_initialized = True
        else:
            print(f"Unknown agent type '{agent_type}', using autopilot")
            self.vehicle.set_autopilot(True)
            return False
        
        if not agent_initialized:
            print("Failed to initialize agent, using autopilot")
            self.vehicle.set_autopilot(True)
            return False
        
        # Set destination with verification
        destination_set = False
        max_attempts = 3
        
        for attempt in range(max_attempts):
            self.agent.set_destination(selected_destination.location)
            
            # Verify destination was set by checking if agent has a plan
            # Wait a few ticks for the agent to compute the route
            for _ in range(5):
                self.world.tick()
            
            # Check if agent is ready to provide control (has computed route)
            test_control = self.agent.run_step()
            if test_control is not None:
                destination_set = True
                print(f"Agent destination set successfully on attempt {attempt + 1}")
                break
            else:
                print(f"Destination setting attempt {attempt + 1} failed, trying another destination")
                if attempt < max_attempts - 1:
                    # Try next destination
                    if len(top_destinations) > attempt + 1:
                        selected_destination, selected_distance = top_destinations[attempt + 1]
                    else:
                        selected_destination, selected_distance = random.choice(suitable_destinations)
                    print(f"Trying new destination {selected_distance:.1f}m away")
        
        if not destination_set:
            print("Failed to set valid destination for agent, using autopilot")
            self.agent = None
            self.vehicle.set_autopilot(True)
            return False
        
        print(f"Agent setup complete: {agent_type}, destination {selected_distance:.1f}m away")
        return True

    def update_agent_control(self):
        """Update vehicle control using intelligent agent"""
        if self.agent is None:
            return  # Using basic autopilot
        
        try:
            # Check if agent reached destination
            if self.agent.done():
                # Set new random destination
                spawn_points = self.world.get_map().get_spawn_points()
                if spawn_points:
                    new_destination = random.choice(spawn_points).location
                    self.agent.set_destination(new_destination)
                    print("Agent reached destination, setting new target")
            
            # Get control from agent
            control = self.agent.run_step()
            control.manual_gear_shift = False
            self.vehicle.apply_control(control)
            
        except Exception as e:
            print(f"Agent control error: {e}")
            # Fallback to autopilot
            self.vehicle.set_autopilot(True)
            self.agent = None

    def setup_npc_agents(self):
        """Setup intelligent control for NPC vehicles to prevent static blocking"""
        if not AGENTS_AVAILABLE:
            # Use basic autopilot for NPCs
            for npc in self.npc_vehicles:
                if npc.is_alive:
                    npc.set_autopilot(True)
            return
        
        # Create varied NPC behaviors to make scene more dynamic
        behaviors = ['cautious', 'normal', 'aggressive']
        
        for i, npc in enumerate(self.npc_vehicles):
            if not npc.is_alive:
                continue
                
            try:
                # Randomly assign behavior types to NPCs
                behavior_choice = random.choice(behaviors)
                
                # Some NPCs use agents, others use autopilot for variety
                if random.random() < 0.7:  # 70% use intelligent agents
                    if random.random() < 0.5:
                        # Use BehaviorAgent
                        agent = BehaviorAgent(npc, behavior=behavior_choice)
                    else:
                        # Use BasicAgent with varied speeds
                        target_speed = random.randint(20, 40)
                        agent = BasicAgent(npc, target_speed=target_speed)
                        agent.follow_speed_limits(True)
                    
                    # Set random destination
                    spawn_points = self.world.get_map().get_spawn_points()
                    if spawn_points:
                        destination = random.choice(spawn_points).location
                        agent.set_destination(destination)
                    
                    # Store agent reference (you might want to add this to class if needed for updates)
                    setattr(npc, '_relight_agent', agent)
                else:
                    # Use basic autopilot
                    npc.set_autopilot(True)
                    setattr(npc, '_relight_agent', None)
                    
            except Exception as e:
                print(f"Failed to setup NPC {i} agent: {e}")
                npc.set_autopilot(True)
                setattr(npc, '_relight_agent', None)

    def update_npc_control(self):
        """Update NPC vehicle control"""
        for npc in self.npc_vehicles:
            if not npc.is_alive:
                continue
                
            agent = getattr(npc, '_relight_agent', None)
            if agent is None:
                continue  # Using autopilot
            
            try:
                # Check if agent reached destination
                if agent.done():
                    # Set new random destination
                    spawn_points = self.world.get_map().get_spawn_points()
                    if spawn_points:
                        new_destination = random.choice(spawn_points).location
                        agent.set_destination(new_destination)
                
                # Apply agent control
                control = agent.run_step()
                control.manual_gear_shift = False
                npc.apply_control(control)
                
            except Exception as e:
                # Fallback to autopilot if agent fails
                npc.set_autopilot(True)
                setattr(npc, '_relight_agent', None)

    def cleanup_actors(self):
        """Clean up all spawned actors including NPCs and agents"""
        # Clean up agent
        self.agent = None
        
        if self.camera_rgb is not None:
            if self.camera_rgb.is_alive:
                self.camera_rgb.stop()
                self.camera_rgb.destroy()
            self.camera_rgb = None
            
        if self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
            
        # Clean up NPC vehicles
        for npc in self.npc_vehicles:
            if npc.is_alive:
                npc.destroy()
        self.npc_vehicles.clear()
            
        print("All actors cleaned up")

    def run_simulation(self, num_scenes=50, images_per_lighting=10):
        """Run simulation across all available maps with dynamic scene counts"""
        self.connect_to_carla()
        
        # Main progress bar for all maps
        total_maps = len(self.available_maps)
        with tqdm(total=total_maps, desc="Processing maps", unit="map") as map_pbar:
            for map_idx, map_name in enumerate(self.available_maps):
                map_pbar.set_description(f"Map: {map_name}")
                
                # Load the current map
                print(f"\n🗺️  Loading map: {map_name}")
                try:
                    self.world = self.client.load_world(map_name)
                    # Set synchronous mode after loading new world
                    self.set_synchronous_mode(synchronous=True, fixed_delta_seconds=0.05)
                    print(f"Successfully loaded map: {map_name}")
                except Exception as e:
                    print(f"Failed to load map {map_name}: {e}")
                    map_pbar.update(1)
                    continue
                
                # Set number of scenes based on map type
                if map_name == 'Town10HD':
                    current_num_scenes = 35
                    print(f"Using {current_num_scenes} scenes for high-detail map {map_name}")
                else:
                    current_num_scenes = 5
                    print(f"Using {current_num_scenes} scenes for map {map_name}")
                
                # Reset world for new map
                self.reset_world()
                
                # Scene progress bar for current map
                with tqdm(total=current_num_scenes, desc=f"Scenes in {map_name}", unit="scene", leave=False) as scene_pbar:
                    for scene_idx in range(current_num_scenes):
                        self.scene_idx = scene_idx
                        scene_pbar.set_description(f"Scene {scene_idx+1:02d}/{current_num_scenes}")
                        
                        # Use simple random spawn point selection
                        spawn_point = self.get_spawn_point_by_strategy('random')
                        
                        # Cleanup previous actors
                        self.cleanup_actors()
                        
                        # Spawn new vehicle and camera
                        if not self.spawn_vehicle(spawn_point):
                            print(f"Failed to spawn vehicle for scene {scene_idx:02d} in {map_name}, skipping...")
                            scene_pbar.update(1)
                            continue
                            
                        # Spawn NPC vehicles for dynamic scene
                        npc_count = self.spawn_npc_vehicles(num_npcs=random.randint(10, 20))
                            
                        if not self.setup_camera():
                            print(f"Failed to setup camera for scene {scene_idx:02d} in {map_name}, skipping...")
                            scene_pbar.update(1)
                            continue
                        
                        # Additional stabilization after spawning all actors
                        print("Stabilizing scene with all actors...")
                        for _ in range(20):
                            self.world.tick()
                        
                        # Capture scene data with map-aware file structure
                        success = self.capture_scene_data(f"{scene_idx+1:02d}", images_per_lighting, map_name)
                        
                        # Update progress bar with scene info
                        scene_pbar.set_postfix({
                            'NPCs': npc_count,
                            'Status': '✓' if success else '✗'
                        })
                        scene_pbar.update(1)
                        
                        if success:
                            print(f"Scene {scene_idx+1:02d} in {map_name} completed with {len(self.lighting_configs)} weather conditions")
                        else:
                            print(f"Scene {scene_idx+1:02d} in {map_name} failed or incomplete")
                
                # Update map progress bar
                map_pbar.set_postfix({
                    'Scenes': current_num_scenes,
                    'Map': map_name
                })
                map_pbar.update(1)
                print(f"✅ Completed map {map_name} with {current_num_scenes} scenes")
                
        self.cleanup_actors()
        total_scenes_processed = sum(35 if map_name == 'Town10HD' else 5 for map_name in self.available_maps)
        print(f"\n🎉 Dataset generation completed! {total_maps} maps processed with {total_scenes_processed} total scenes.")
        print(f"📁 Images saved in structure: {self.output_dir}/{{map_name}}/{{scene_num}}/{{weather}}/")

    def save_image(self, image, lighting_name, scene_name, image_idx, map_name):
        """保存图像回调函数 - 新的文件夹结构：{map_name}/{scene_num}/{weather}/"""
        # 创建目录结构：{map_name}/{scene_num}/{weather}/
        scene_dir = os.path.join(self.output_dir, map_name, f"scene_{self.scene_idx:03d}")
        lighting_dir = os.path.join(scene_dir, lighting_name)
        os.makedirs(lighting_dir, exist_ok=True)
        
        # 保存图像，文件名只包含数字
        filename = f"{image_idx:06d}.png"
        filepath = os.path.join(lighting_dir, filename)
        image.save_to_disk(filepath)

    def capture_single_image(self, lighting_name, scene_name, image_idx, map_name):
        """Capture a single image with improved synchronization"""
        image_captured = False
        captured_image = None
        
        def image_callback(image):
            nonlocal image_captured, captured_image
            if not image_captured:
                captured_image = image
                image_captured = True
        
        # Start listening for the image
        self.camera_rgb.listen(image_callback)
        
        # Multiple ticks to ensure image is captured
        max_attempts = 10
        for attempt in range(max_attempts):
            self.world.tick()
            if image_captured:
                break
        
        # Stop listening
        self.camera_rgb.stop()
        
        if image_captured and captured_image:
            self.save_image(captured_image, lighting_name, scene_name, image_idx, map_name)
            return True
        else:
            print(f"    ✗ Failed to capture image after {max_attempts} attempts")
            return False

    def spawn_vehicle(self, spawn_point=None):
        """Generate vehicle without autopilot initially"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        
        if not vehicle_bps:
            print("No vehicle blueprints found!")
            return False
            
        vehicle_bp = random.choice(vehicle_bps)
        
        # Randomize vehicle color
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)
        
        # Get spawn point
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("No spawn points available!")
                return False
            spawn_point = random.choice(spawn_points)
        
        # Try to spawn vehicle
        max_attempts = 10
        for attempt in range(max_attempts):
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                print(f"Vehicle spawned successfully: {self.vehicle.type_id}")
                
                # Don't enable autopilot immediately - will be controlled during recording/replay
                
                # Wait for vehicle to be fully spawned
                if self.world.get_settings().synchronous_mode:
                    for _ in range(5):
                        self.world.tick()
                
                return True
            else:
                # Try different spawn point
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points)
                print(f"Spawn attempt {attempt + 1} failed, trying different location...")
        
        print(f"Failed to spawn vehicle after {max_attempts} attempts")
        return False

    def spawn_npc_vehicles(self, num_npcs=15):
        """Spawn NPC vehicles for dynamic scenes"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Shuffle spawn points to get random positions
        random.shuffle(spawn_points)
        
        spawned_count = 0
        for i, spawn_point in enumerate(spawn_points[:num_npcs * 2]):  # Try more points than needed
            if spawned_count >= num_npcs:
                break
                
            vehicle_bp = random.choice(vehicle_bps)
            
            # Randomize vehicle color
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
            
            # Try to spawn NPC vehicle
            npc = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if npc is not None:
                self.npc_vehicles.append(npc)
                # Don't enable autopilot immediately - will be controlled during recording/replay
                spawned_count += 1
                print(f'Created NPC vehicle: {npc.type_id}')
        
        print(f"Spawned {spawned_count} NPC vehicles")
        return spawned_count > 0

    def setup_camera(self):
        """Setup RGB camera attached to vehicle"""
        # Get camera blueprint
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Set camera parameters
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')

        # Enable high quality rendering
        camera_bp.set_attribute('enable_postprocess_effects', 'true')
        camera_bp.set_attribute('gamma', '2.2')
        
        vehicle_bbox = self.vehicle.bounding_box
        vehicle_length = vehicle_bbox.extent.x # * 2 
        camera_x = vehicle_length + 1.0
        # Set camera position (attached to vehicle) - similar to tutorial but for RGB
        camera_transform = carla.Transform(
            carla.Location(x=camera_x, z=2.4),
            carla.Rotation(pitch=0.0)
        )
        
        # Spawn camera
        self.camera_rgb = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        print(f"RGB camera created: {self.camera_rgb.type_id}")
        return True

    def capture_scene_data(self, scene_strategy, num_images_per_lighting=10, map_name=""):
        """Capture scene data using trajectory recording and replay for consistency"""
        print(f"\nStarting scene {self.scene_idx}: {scene_strategy}")
        
        # First, record the trajectory under default lighting
        print("Step 1: Recording trajectory under default lighting...")
        default_weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            sun_altitude_angle=70.0,
            sun_azimuth_angle=0.0
        )
        self.world.set_weather(default_weather)
        
        # Wait for weather to stabilize
        for _ in range(10):
            self.world.tick()
        
        # Enable autopilot for natural movement during recording
        self.vehicle.set_autopilot(True)
        for npc in self.npc_vehicles:
            if npc.is_alive:
                npc.set_autopilot(True)
        
        # Record trajectory
        recording_success = self.record_scene_trajectory(duration_seconds=75)
        
        if not recording_success:
            print("Failed to record trajectory, skipping scene...")
            return False
        
        print("Step 2: Replaying scene under different lighting conditions...")
        
        # Select 10 random lighting configurations from all 14 available
        selected_lighting_configs = random.sample(self.lighting_configs, min(10, len(self.lighting_configs)))
        print(f"Selected {len(selected_lighting_configs)} lighting conditions from {len(self.lighting_configs)} available:")
        for config in selected_lighting_configs:
            print(f"  - {config['name']}")
        
        # Now replay the same trajectory under selected lighting conditions with progress bar
        total_images_expected = len(selected_lighting_configs) * num_images_per_lighting
        # breakpoint()
        with tqdm(total=len(selected_lighting_configs), desc="Lighting conditions", unit="condition") as lighting_pbar:
            for i, lighting_config in enumerate(selected_lighting_configs):
                lighting_pbar.set_description(f"Lighting: {lighting_config['name'][:15]}")
                
                success = self.replay_scene_trajectory(lighting_config, num_images_per_lighting, map_name)
                lighting_pbar.set_postfix({
                    'Status': '✓' if success else '✗',
                    'Images': f"{num_images_per_lighting}/light"
                })
                lighting_pbar.update(1)
                
                if not success:
                    print(f"    Failed to replay scene with {lighting_config['name']}")
        
        print(f"Scene {self.scene_idx} completed with consistent trajectory replay")
        print(f"Expected total images: {total_images_expected}")
        return True

    def connect_to_carla(self):
        """Connect to CARLA server and configure world"""
        self.world = self.client.get_world()
        print(f"Connected to CARLA server, current map: {self.world.get_map().name}")
        # breakpoint()
        # Set synchronous mode with recommended settings for data collection
        self.set_synchronous_mode(synchronous=True, fixed_delta_seconds=0.05)
        
        return True

    def apply_lighting_config(self, config):
        """应用光照配置到世界"""
        if self.world is None:
            print("World not initialized, cannot apply lighting config")
            return False
            
        weather = carla.WeatherParameters(
            cloudiness=float(config['cloudiness']),
            precipitation=float(config['precipitation']),
            sun_altitude_angle=float(config['sun_altitude_angle']),
            sun_azimuth_angle=float(config['sun_azimuth_angle'])
        )
        
        self.world.set_weather(weather)
        print(f"Applied lighting config: {config['name']}")
        
        # Wait for weather to take effect
        if self.world.get_settings().synchronous_mode:
            for _ in range(3):
                self.world.tick()
        else:
            time.sleep(0.5)
                
        return True
    
    def initialize_world(self):
        """初始化世界连接"""
        self.world = self.client.get_world()
        print("World initialized successfully")
        return True
    
    def test_lighting_configs(self):
        """测试所有光照配置"""
        self.initialize_world()
            
        print("Testing lighting configurations...")
        print(f"Current map: {self.world.get_map().name}")
        
        for i, config in enumerate(self.lighting_configs):
            print(f"\n--- Testing config {i+1}/{len(self.lighting_configs)}: {config['name']} ---")
            
            success = self.apply_lighting_config(config)
            if success:
                print(f"  ✓ Successfully applied {config['name']}")
                time.sleep(2.0)
            else:
                print(f"  ✗ Failed to apply {config['name']}")

    def get_spawn_point_by_strategy(self, strategy):
        """Simplified random spawn point selection"""
        spawn_points = self.world.get_map().get_spawn_points()
        return random.choice(spawn_points)

def main():
    parser = argparse.ArgumentParser(description='CARLA Relight Dataset Generation Tool')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server address')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--output-dir', default='./carla_relight_data', help='Output directory')
    parser.add_argument('--num-scenes', type=int, default=20, help='Number of scenes to collect')
    parser.add_argument('--images-per-lighting', type=int, default=30, help='Images per lighting condition')
    
    args = parser.parse_args()
    
    # Create simulator instance
    simulator = CarlaRelightSimulator(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir
    )
    
    # Run simulation with moving vehicles
    simulator.run_simulation(
        num_scenes=args.num_scenes,
        images_per_lighting=args.images_per_lighting
    )

if __name__ == '__main__':
    main()
