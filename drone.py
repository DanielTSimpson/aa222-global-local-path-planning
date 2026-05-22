"""
Drone module for Dec-POMDP multi-agent system
Implements intelligent decision-making for science search and extinguish
"""

import numpy as np
from copy import deepcopy
from environment import SearchEnv
from belief import Belief
from sensors import OnlineSensor
from local_planner import *
import config as cfg


class Drone:
    """
    Dec-POMDP Agent with belief state and value-based decision making.
    """

    def __init__(self, environment: SearchEnv):
        self.drone_id = 0
        self.window_size = cfg.OBSERVATION_WINDOW_SIZE
        self.env = environment

        self.position = np.array([environment.grid_size - 2, environment.grid_size - 2])
        self.budget = cfg.MAX_BUDGET_PER_DRONE
        self.belief_state = Belief(self.env.grid_size)
        self.lookahead_depth = cfg.LOOKAHEAD_DEPTH

        self.time = 0
        self.visited_cells = set()
        self.last_action = None
        self.science_found = False
        self.drifted = False
        self.history = []
        self.stuck_count = 0

        # POMDP params
        self.gamma = 0.0
        self.exploration_bonus = 0.0
        self.movement_cost = 0.0
        self.time_cost = 0.0

        # Sensor params
        self.sensor = OnlineSensor(
            view_depth=cfg.SENSOR_VIEW_DEPTH,
            view_angle=cfg.SENSOR_VIEW_ANGLE,
            false_negative_rate=cfg.SENSOR_FALSE_NEGATIVE_RATE,
            false_positive_rate=cfg.SENSOR_FALSE_POSITIVE_RATE,
        )

        self.heading = 0.0
        self.known_small_obstacles = set()
        self.known_science = set()
        
        self.known_small_science = set()
        self.small_science_collected_value = 0
        
        self.current_visible_cells = set()

        self.science_found = self.observe()
        if not self.history:
            self.history.append(self.state)
            
        self.local_optimizer = SimulatedAnnealingOptimizer(horizon = 5, iterations = 100, initial_temp = 10.0, cooling = 0.95, weights = cfg.LOCAL_PLANNER_WEIGHTS)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def state(self):
        return [self.x, self.y, self.science_found]

    def action(self, action):
        """Execute action and update state."""
        self.last_action = action

        step_cost = self.time_cost
        if action in [2, 3, 4, 5, 6, 7, 8, 9]:
            step_cost += self.movement_cost
        self.budget -= step_cost

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

        self.drifted = False

        if cfg.ENABLE_WIND and np.random.random() < abs(self.env.wind_speed):
            cos_wind = np.cos(self.env.wind_direction)
            sin_wind = np.sin(self.env.wind_direction)

            dx = int(np.sign(cos_wind)) if np.random.random() < abs(cos_wind) else 0
            dy = int(np.sign(sin_wind)) if np.random.random() < abs(sin_wind) else 0

            if dx != 0 or dy != 0:
                self.drifted = True
                x = max(0, min(self.env.grid_size - 1, x + dx))
                y = max(0, min(self.env.grid_size - 1, y + dy))

            if self.env.obstacle_grid[x, y] == 1:
                x, y = prev_position

        candidate = np.array([x, y])

        if self.env.is_obstacle(candidate[0], candidate[1]):
            candidate = prev_position

        self.position = candidate
        
        if cfg.SMALL_SCIENCE_ENABLED:
            current_cell = tuple(self.position)
            if current_cell in self.env.small_science:
                if current_cell not in self.env.collected_small_science:
                    value = self.env.small_science[current_cell]
                    self.small_science_collected_value += value
                    self.env.collected_small_science.add(current_cell)
                    self.known_small_science.discard(current_cell)

                    print(f"Drone {self.drone_id} collected small science at {current_cell} worth {value}")

        if np.array_equal(self.position, prev_position):
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if not self.env.science_collected:
            self.science_found = self.observe()

        self.history.append(self.state)
        self.time += 1

    def observe(self):
        """Update belief based on online sensor observation."""
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

        self.belief_state.update_from_observation(
            self.position,
            self.window_size,
            science_observed,
        )

        if science_observed:
            print(f"Drone {self.drone_id} found science objective!")

        return science_observed

    def create_telemetry_packet(self):
        """Creates telemetry packet with belief state."""
        packet = {
            "sender_id": self.drone_id,
            "timestamp": self.time,
            "position": self.position.copy(),
            "belief_state": self.belief_state.copy(),
            "visited_cells": self.visited_cells.copy(),
            "history": deepcopy(self.history),
        }
        return packet

    def receive_telemetry(self, packet, communication_noise=0.1):
        """Receive and merge belief states."""
        other_visited = packet["visited_cells"]
        self.visited_cells.update(other_visited)

        if "belief_state" in packet:
            self.belief_state.merge(packet["belief_state"])

    def _get_best_value(self, belief, position, visited, depth):
        """
        Recursive helper to calculate the best Q-value from a given state
        with limited lookahead.
        """
        if depth == 0:
            return 0.0

        max_q = -float("inf")
        moves = {
            2: (0, 1),
            3: (0, -1),
            4: (-1, 0),
            5: (1, 0),
            6: (1, 1),
            7: (-1, 1),
            8: (1, -1),
            9: (-1, -1),
        }

        current_entropy = belief.get_entropy()

        for _, (dx, dy) in moves.items():
            nx = max(0, min(self.env.grid_size - 1, position[0] + dx))
            ny = max(0, min(self.env.grid_size - 1, position[1] + dy))

            half = self.window_size // 2
            x_min = max(0, nx - half)
            x_max = min(self.env.grid_size, nx + half + 1)
            y_min = max(0, ny - half)
            y_max = min(self.env.grid_size, ny + half + 1)

            new_cells = sum(
                1
                for r in range(x_min, x_max)
                for c in range(y_min, y_max)
                if (r, c) not in visited
            )

            bonus = self.exploration_bonus if new_cells > 0 else 0
            reward_action = bonus - self.movement_cost - self.time_cost

            prob_see_science = np.sum(belief.belief_grid[x_min:x_max, y_min:y_max])
            prob_see_nothing = 1.0 - prob_see_science

            gain_see_science = current_entropy

            temp_belief = belief.copy()
            temp_belief.update_from_observation(
                (nx, ny),
                self.window_size,
                science_found=False,
            )
            gain_see_nothing = current_entropy - temp_belief.get_entropy()

            future_val = 0.0
            if depth > 1 and prob_see_nothing > 0:
                new_visited = visited.copy()
                for r in range(x_min, x_max):
                    for c in range(y_min, y_max):
                        new_visited.add((r, c))

                future_val = self._get_best_value(
                    temp_belief,
                    (nx, ny),
                    new_visited,
                    depth - 1,
                )

            val_nothing = gain_see_nothing + self.gamma * future_val
            q = reward_action + self.gamma * (
                prob_see_science * gain_see_science
                + prob_see_nothing * val_nothing
            )

            if q > max_q:
                max_q = q

        return max_q

    def decide_action_pomdp(self):
        """
        Lookahead POMDP planning.
        Returns the action index with the highest Q-value.
        """
        
        # working in our local optimizer
        if len(self.known_small_obstacles) > 0 or len(self.known_small_science) > 0:
            return self.local_optimizer.choose_action(self, self.env)

        # If science has been seen, move directly toward it.
        if len(self.known_science) > 0:
            sx, sy = next(iter(self.known_science))

            if self.x == sx and self.y == sy:
                return 1  # Collect

            dx = sx - self.x
            dy = sy - self.y

            if dx > 0 and dy > 0:
                return 6  # Up-Right
            elif dx < 0 and dy > 0:
                return 7  # Up-Left
            elif dx > 0 and dy < 0:
                return 8  # Down-Right
            elif dx < 0 and dy < 0:
                return 9  # Down-Left
            elif dy > 0:
                return 2  # Up
            elif dy < 0:
                return 3  # Down
            elif dx < 0:
                return 4  # Left
            elif dx > 0:
                return 5  # Right

        best_actions = [0]
        max_q_value = -float("inf")

        current_entropy = self.belief_state.get_entropy()

        moves = {
            2: (0, 1),
            3: (0, -1),
            4: (-1, 0),
            5: (1, 0),
            6: (1, 1),
            7: (-1, 1),
            8: (1, -1),
            9: (-1, -1),
        }

        for action_idx, (dx, dy) in moves.items():
            nx = max(0, min(self.env.grid_size - 1, self.x + dx))
            ny = max(0, min(self.env.grid_size - 1, self.y + dy))

            if self.env.is_obstacle(nx, ny):
                continue

            half = self.window_size // 2
            x_min = max(0, nx - half)
            x_max = min(self.env.grid_size, nx + half + 1)
            y_min = max(0, ny - half)
            y_max = min(self.env.grid_size, ny + half + 1)

            new_cells = sum(
                1
                for r in range(x_min, x_max)
                for c in range(y_min, y_max)
                if (r, c) not in self.visited_cells
            )

            bonus = self.exploration_bonus if new_cells > 0 else 0
            reward_action = bonus - self.movement_cost - self.time_cost

            prob_see_science = np.sum(
                self.belief_state.belief_grid[x_min:x_max, y_min:y_max]
            )
            prob_see_nothing = 1.0 - prob_see_science

            gain_see_science = current_entropy

            temp_belief = self.belief_state.copy()
            temp_belief.update_from_observation(
                (nx, ny),
                self.window_size,
                science_found=False,
            )
            gain_see_nothing = current_entropy - temp_belief.get_entropy()

            future_val = 0.0
            if self.lookahead_depth > 1 and prob_see_nothing > 0:
                new_visited = self.visited_cells.copy()

                for r in range(x_min, x_max):
                    for c in range(y_min, y_max):
                        new_visited.add((r, c))

                future_val = self._get_best_value(
                    temp_belief,
                    (nx, ny),
                    new_visited,
                    self.lookahead_depth - 1,
                )

            val_nothing = gain_see_nothing + self.gamma * future_val
            q_value = reward_action + self.gamma * (
                prob_see_science * gain_see_science
                + prob_see_nothing * val_nothing
            )

            if q_value > max_q_value:
                max_q_value = q_value
                best_actions = [action_idx]
            elif np.isclose(q_value, max_q_value):
                best_actions.append(action_idx)

        return int(np.random.choice(best_actions))