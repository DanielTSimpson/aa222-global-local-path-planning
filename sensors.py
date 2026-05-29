import numpy as np
import config as cfg

class OnlineSensor: # The agent's frustum-based camera
  
    def __init__(self, view_depth = 4, view_angle = np.pi/2, false_negative_rate = 0.05, false_positive_rate = 0.01):
        self.view_depth = view_depth
        self.view_angle = view_angle
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
    

    def get_visible_cells(self, drone_pos, heading, grid_size = cfg.GRID_SIZE):
        r0, c0 = drone_pos # r0 is the row our drone is in (its y coord) and c0 is the column (its x coord)
        visible_cells = []

        for r in range(max(0, r0 - self.view_depth), min(grid_size, r0 + self.view_depth + 1)):
            for c in range(max(0, c0 - self.view_depth), min(grid_size, c0 + self.view_depth + 1)):
                dr = r - r0 # relative row coordinate wrt the agent's row position
                dc = c - c0 # relative col coordinate wrt the agent's col position

                dist = np.sqrt(dr**2 + dc**2)
                if dist == 0 or dist > self.view_depth: continue
                angle = np.arctan2(dc, dr)

                # compare the angle from the drone to the cell with the drone's own heading
                angle_error = self.wrap_angle(angle - heading)

                if abs(angle_error) <= self.view_angle / 2:
                    visible_cells.append((r, c))

        return visible_cells
    

    def observe(self, drone_pos, heading, environment):
        visible_cells = self.get_visible_cells(drone_pos, heading, grid_size = environment.grid_size)
        observations = {"visible_cells": visible_cells, "detected_obstacles": [], "detected_science": [], "detected_small_science": [], "detected_small_science_scores": {}, "missed_cells": []}

        for cell in visible_cells:
            r, c = cell

            ## Obstacle checks with false positive and false negative events
            if environment.is_obstacle(r, c):
                if np.random.random() > self.false_negative_rate:
                    observations["detected_obstacles"].append(cell)
                else:
                    observations["missed_cells"].append(cell)
            else:
                if np.random.random() < self.false_positive_rate:
                    observations["detected_obstacles"].append(cell)
                
            ## Science observation with false negative events
            if np.array_equal(np.array([r, c]), environment.science_pos):
                if np.random.random() > self.false_negative_rate:
                    observations["detected_science"].append(cell)
                else:
                    observations["missed_cells"].append(cell)
                    
            ## Small, local science observation with false negative events
            if cfg.SMALL_SCIENCE_ENABLED:
                if cell in environment.small_science:
                    if cell not in environment.collected_small_science:
                        if np.random.random() > self.false_negative_rate:
                            observations["detected_small_science"].append(cell)
                        else:
                            observations["missed_cells"].append(cell)

        return observations
    
    
    @staticmethod
    def wrap_angle(angle): # Wraps angle from [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi