"""
Drone module for Dec-POMDP multi-agent system
Implements intelligent decision-making for science search and extinguish
"""

import numpy as np
from environment import SearchEnv
from sensors import OnlineSensor
from local_planner import *
import config as cfg


class Drone:
    def __init__(self, environment: SearchEnv):
        self.env = environment

        self.position = np.array([environment.grid_size - 2, environment.grid_size - 2])
        self.heading = 0.0
        self.budget = cfg.MAX_BUDGET

        self.time = 0
        self.visited_cells = set()
        self.last_action = None
        self.science_found = False
        self.history = []
        self.stuck_count = 0

        # Sensor params
        self.sensor = OnlineSensor(
            view_depth=cfg.SENSOR_VIEW_DEPTH,
            view_angle=cfg.SENSOR_VIEW_ANGLE,
            false_negative_rate=cfg.SENSOR_FALSE_NEGATIVE_RATE,
            false_positive_rate=cfg.SENSOR_FALSE_POSITIVE_RATE,
        )

        self.known_small_obstacles = set()
        self.known_science = set()
        
        self.known_small_science = set()
        self.known_small_science_scores = {}
        self.small_science_collected_value = 0
        
        self.current_visible_cells = set()

        # For free, observe the surrounding area at spawn
        aggregated_observations = {
            "visible_cells": [],
            "detected_obstacles": [],
            "detected_science": [],
            "detected_small_science": [],
            "detected_small_science_scores": {},
            "missed_cells": [],
        }
        for i in range(8):
            single_observation = self.sensor.observe(
                self.position,
                self.heading + i*np.pi/4,
                self.env
            )
            for key, value in single_observation.items():
                if key == "detected_small_science_scores":
                    aggregated_observations[key].update(value)
                else:
                    aggregated_observations[key].extend(value)
        self.science_found = self.observe(aggregated_observations)
        
        self.local_optimizer = make_local_planner(cfg.LOCAL_PLANNER_TYPE) 

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def action(self, action):
        """Execute action and update state."""
        self.last_action = action

        prev_position = self.position.copy()
        x = self.x
        y = self.y

        if action == 1:  # Collect science
            if np.array_equal(self.position, self.env.science_pos):
                self.env.science_collected = True

        elif action == 2:  # Up
            y = min(self.env.grid_size - 1, self.y + 1)
            self.heading = np.pi / 2

        elif action == 3:  # Down
            y = max(0, self.y - 1)
            self.heading = -np.pi / 2

        elif action == 4:  # Left
            x = max(0, self.x - 1)
            self.heading = np.pi

        elif action == 5:  # Right
            x = min(self.env.grid_size - 1, self.x + 1)
            self.heading = 0.0

        elif action == 6:  # Up-Right
            x = min(self.env.grid_size - 1, self.x + 1)
            y = min(self.env.grid_size - 1, self.y + 1)
            self.heading = np.pi / 4

        elif action == 7:  # Up-Left
            x = max(0, self.x - 1)
            y = min(self.env.grid_size - 1, self.y + 1)
            self.heading = 3 * np.pi / 4

        elif action == 8:  # Down-Right
            x = min(self.env.grid_size - 1, self.x + 1)
            y = max(0, self.y - 1)
            self.heading = -np.pi / 4

        elif action == 9:  # Down-Left
            x = max(0, self.x - 1)
            y = max(0, self.y - 1)
            self.heading = -3 * np.pi / 4

        candidate = np.array([x, y])

        if self.env.is_obstacle(candidate[0], candidate[1]):
            candidate = prev_position

        self.position = candidate
        
        step_cost = self.time_cost
        if action in cfg.MOVES:
            step_cost += self.movement_cost
        self.budget -= step_cost

        if cfg.SMALL_SCIENCE_ENABLED:
            current_cell = tuple(self.position)
            if current_cell in self.env.small_science:
                if current_cell not in self.env.collected_small_science:
                    value = self.env.small_science[current_cell]
                    self.small_science_collected_value += value
                    self.env.collected_small_science.add(current_cell)
                    self.known_small_science.discard(current_cell)
                    self.known_small_science_scores.pop(current_cell, None)
                    if cfg.VERBOSE_LOGGING: print(f"Drone collected small science at {current_cell} worth {value}")

        if np.array_equal(self.position, prev_position):
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if not self.env.science_collected:
            self.science_found = self.observe()

        self.time += 1

    def observe(self, observations = None):
        """Update belief based on online sensor observation."""
        if observations == None:
            observations = self.sensor.observe(
                drone_pos=self.position,
                heading=self.heading,
                environment=self.env,
            )
        
        self.current_visible_cells = set(observations["visible_cells"])

        for cell in observations["visible_cells"]:
            self.visited_cells.add(cell)

        for obstacle in observations["detected_obstacles"]:
            self.known_small_obstacles.add(obstacle)

        for science_cell in observations["detected_science"]:
            self.known_science.add(science_cell)
            
        for small_science_cell in observations["detected_small_science"]:
            self.known_small_science.add(small_science_cell)

        science_observed = len(observations["detected_science"]) > 0

        if science_observed and cfg.VERBOSE_LOGGING:
            print("Drone found science objective!")

        return science_observed

    def local_optimizer_needed(self, next_global_action = None):
        # checks to see if something is in the global path or if there's something worth deviating for, and then returns if a local optimizer is neede
        
        # first, we see if there's a small science objective that's been detected
        if cfg.SMALL_SCIENCE_ENABLED and len(self.known_small_science) > 0:
            return True
        
        # then we check to see if the drone is repeatedly failing to move
        if self.stuck_count > 0:
            return True
        
        # lastly, we check if the next global path step points into a known small obstacle
        if next_global_action is not None:
            if next_global_action not in cfg.MOVES:
                return False
            
            dx, dy = cfg.MOVES[next_global_action]
            
            next_cell = (self.x + dx, self.y + dy)
            
            if next_cell in self.known_small_obstacles:
                return True
            
        return False